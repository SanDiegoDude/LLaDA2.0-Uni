"""
LLaDA-2.0-Uni — Gradio UI.

Wraps the CLI's plumbing in a persistent-state Gradio app so model + decoder +
VAE + image tokenizer all stay resident across generations. This eliminates the
~90 s of reload overhead you pay on every CLI invocation and makes the decoder
replay tab genuinely fast (<15 s per re-decode once warm).

Launch:

    source .venv/bin/activate
    python scripts/ui.py --quant nf4 --host 0.0.0.0 --port 7860

For 24 GB consumer cards (RTX 4090, 3090) add ``--low_vram`` — the LLM and
decoder are then swapped on/off the GPU between stages, keeping peak VRAM at
~14 GB at the cost of ~75 s of LLM reload between generations. The "Replay
decoder" tab is unaffected and stays fast because only the decoder is resident
during that loop.

Design notes:

- NF4 backbone loads eagerly at startup (~75 s) since most users will want
  to generate immediately and the model dominates memory.
- Decoder ``diff_model`` is loaded lazily on first decode and then cached
  per mode (turbo / normal). Loading the turbo variant eagerly along with
  the LLM would be an extra ~12 GB of VRAM for a one-time save of ~30 s;
  lazy feels like the better default.
- VAE + SigVQ are small (<1 GB combined) and cached after first use.
- Image tokenizer (encoder) is only needed for edit / mmu; also lazy.
- The transport sampler is re-constructed each decode (cheap).

The pipeline is process-global so Gradio's stateless request model works
fine without passing heavy objects through `gr.State`.
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# The LLaDA-2 tokenizer ships with the legacy (training-matched) Mistral
# tekken regex. transformers 4.55+ prints a loud warning suggesting we set
# fix_mistral_regex=True, but doing so would mismatch training — keep the
# legacy behaviour and mute the warning.
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=r".*incorrect regex pattern.*")

# Locked decoder scale. Anything other than 2 (the training-time value)
# produces visibly degraded output — mushy edges, wrong aspect handling,
# or outright artifacts. Keeping the dial exposed to users was a footgun.
_FIXED_RES_MULT = 2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import CLI plumbing so we stay in sync with any quant/model-path changes.
from scripts.llada import (  # noqa: E402
    MODEL_PATHS,
    _build_fp8_config,
    _ensure_model_path,
    _image_token_offset,
    _is_quantized,
)


# ---------------------------------------------------------------------------
# Persistent pipeline
# ---------------------------------------------------------------------------
@dataclass
class Pipeline:
    """Process-global cache of every heavy weight we might need.

    Call ``.load_llm()`` once at startup; the other getters populate lazily
    on first use.
    """

    model_path: str
    quant: str
    device: str = "cuda"
    # When True, swap LLM ↔ decoder on GPU so the two never co-reside.
    # Target use: 24 GB consumer cards (4090, 3090) where LLM(9.5) + decoder(12.3)
    # would otherwise clip. Pays ~75 s LLM reload between every generation, but
    # decoder-replay stays instant once warm.
    low_vram: bool = False
    # "Load on demand": unload ALL components after each request completes,
    # so the process sits at ~0 GB VRAM between requests. Pays the full
    # reload cost every call. Useful for leaving the API server running
    # long-term without hogging GPU memory.
    lod: bool = False

    llm: object | None = None
    tokenizer: object | None = None
    sigvq: object | None = None
    vae: object | None = None
    image_tokenizer: object | None = None
    decoder_models: dict[str, object] = field(default_factory=dict)  # mode -> diff_model
    decoder_cfgs: dict[str, dict] = field(default_factory=dict)      # mode -> cfg.json
    last_vq: dict | None = None  # cache of last generated/edited VQ tokens

    # --- load helpers ---
    def load_llm(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.llm is not None:
            return
        print(f"[LLM] Loading {self.quant} from {self.model_path}")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        kwargs = dict(
            trust_remote_code=True,
            device_map={"": self.device},
            torch_dtype=torch.bfloat16,
        )
        if self.quant == "fp8":
            kwargs["quantization_config"] = _build_fp8_config()
        m = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs).eval()
        if self.quant == "bf16" and not _is_quantized(self.model_path):
            m = m.to(torch.bfloat16)
        m.tokenizer = self.tokenizer
        self.llm = m
        print(f"[LLM] ready in {time.time() - t0:.1f}s, "
              f"CUDA={torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def load_sigvq(self) -> None:
        from decoder.sigvq import SigVQ

        if self.sigvq is not None:
            return
        print("[SigVQ] loading")
        t0 = time.time()
        m = SigVQ(vocab_size=16384, inner_dim=4096).to(
            self.device, dtype=torch.bfloat16,
        )
        m.load_state_dict(torch.load(
            os.path.join(self.model_path, "image_tokenizer", "sigvq_embedding.pt"),
            map_location=self.device, weights_only=True,
        ))
        self.sigvq = m.eval()
        print(f"[SigVQ] ready in {time.time() - t0:.1f}s")

    def load_decoder(self, mode: str) -> None:
        """mode ∈ {"turbo", "normal"}. Caches both architecture cfg + weights."""
        from safetensors.torch import load_file

        from decoder.decoder_model import ZImageTransformer2DModel

        if mode in self.decoder_models:
            return
        sub = "decoder-turbo" if mode == "turbo" else "decoder"
        print(f"[Decoder:{mode}] loading {sub}/")
        t0 = time.time()
        cfg = json.load(open(os.path.join(self.model_path, sub, "config.json")))
        cfg["axes_lens"] = [32768, 1024, 1024]
        cfg["cap_feat_dim"] = 4096
        with torch.device("meta"):
            m = ZImageTransformer2DModel(**cfg)
        ckpt = os.path.join(self.model_path, sub, "decoder_model.safetensors")
        m.load_state_dict(
            load_file(ckpt, device=str(self.device)), assign=True,
        )
        self.decoder_models[mode] = m.to(dtype=torch.bfloat16).eval()
        self.decoder_cfgs[mode] = cfg
        print(f"[Decoder:{mode}] ready in {time.time() - t0:.1f}s, "
              f"CUDA={torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def load_vae(self) -> None:
        from diffusers import AutoencoderKL

        if self.vae is not None:
            return
        print("[VAE] loading")
        t0 = time.time()
        self.vae = AutoencoderKL.from_pretrained(
            os.path.join(self.model_path, "vae"), torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        print(f"[VAE] ready in {time.time() - t0:.1f}s")

    def load_image_tokenizer(self) -> None:
        from encoder.image_tokenizer import ImageTokenizer

        if self.image_tokenizer is not None:
            return
        print("[ImageTokenizer] loading")
        t0 = time.time()
        self.image_tokenizer = ImageTokenizer(
            model_path=self.model_path, device=self.device, dtype=torch.bfloat16,
        )
        print(f"[ImageTokenizer] ready in {time.time() - t0:.1f}s")

    # --- unload helpers (used by low_vram mode) ---
    def unload_llm(self) -> None:
        if self.llm is None:
            return
        print("[LLM] unloading")
        del self.llm
        self.llm = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def unload_decoder(self, mode: str | None = None) -> None:
        """Drop one decoder mode (or all if mode is None)."""
        if mode is None:
            modes = list(self.decoder_models.keys())
        else:
            modes = [mode] if mode in self.decoder_models else []
        for m in modes:
            print(f"[Decoder:{m}] unloading")
            del self.decoder_models[m]
            self.decoder_cfgs.pop(m, None)
        if modes:
            gc.collect()
            torch.cuda.empty_cache()

    def unload_image_tokenizer(self) -> None:
        if self.image_tokenizer is None:
            return
        print("[ImageTokenizer] unloading")
        del self.image_tokenizer
        self.image_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def unload_sigvq(self) -> None:
        if self.sigvq is None:
            return
        print("[SigVQ] unloading")
        del self.sigvq
        self.sigvq = None
        gc.collect()
        torch.cuda.empty_cache()

    def unload_vae(self) -> None:
        if self.vae is None:
            return
        print("[VAE] unloading")
        del self.vae
        self.vae = None
        gc.collect()
        torch.cuda.empty_cache()

    def unload_all(self) -> None:
        """Drop everything from GPU. Used by --lod after each request."""
        self.unload_llm()
        self.unload_decoder()
        self.unload_image_tokenizer()
        self.unload_sigvq()
        self.unload_vae()

    def finalize_request(self) -> None:
        """Run at the end of every UI / API request (in a finally block).

        Currently a no-op unless ``--lod`` is on, in which case it flushes
        every weight back out of GPU memory so the process idles near 0 GB.
        """
        if self.lod:
            print("[lod] request finished, unloading everything")
            t0 = time.time()
            self.unload_all()
            print(f"[lod] idle after {time.time() - t0:.1f}s, "
                  f"CUDA={torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- decode primitives ---
    @torch.inference_mode()
    def decode(
        self,
        token_ids: list[int],
        h: int,
        w: int,
        decoder_mode: str = "turbo",
        num_steps: int | None = None,
        resolution_multiplier: int = 2,
        seed: int = 42,
        progress_cb=None,
    ) -> Image.Image:
        from torchvision.transforms.functional import to_pil_image

        from decoder.transport import create_transport, Sampler

        # Make room for the decoder before loading it. The LLM + decoder+
        # VAE combo pushes ~22 GB of model weights alone, which is too close
        # to the limit on 24 GB consumer GPUs. Both low_vram and lod trigger
        # this swap (lod unloads everything AFTER the request, low_vram just
        # during transitions).
        if self.low_vram or self.lod:
            self.unload_llm()
            self.unload_image_tokenizer()

        self.load_sigvq()
        self.load_decoder(decoder_mode)
        self.load_vae()

        diff_model = self.decoder_models[decoder_mode]
        cfg = self.decoder_cfgs[decoder_mode]
        if num_steps is None:
            num_steps = 8 if decoder_mode == "turbo" else 50

        # Stage 1: SigVQ → semantic features
        th = h * 16 * resolution_multiplier
        tw = w * 16 * resolution_multiplier
        tok = torch.tensor(token_ids).view(1, 1, h, w).float().to(self.device)
        up = F.interpolate(tok, scale_factor=2, mode="nearest").long().view(1, -1)
        cap_pos = [self.sigvq(up).squeeze(0)]
        cap_neg = [torch.zeros_like(cap_pos[0])]

        # Stage 2: diffusion ODE
        g = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn([1, 16, 1, 2 * (th // 16), 2 * (tw // 16)],
                        device=self.device, generator=g)

        n = 1
        doubled = cap_pos + cap_neg
        patch = cfg.get("all_patch_size", (2,))[0]
        fpatch = cfg.get("all_f_patch_size", (1,))[0]
        # Our decode.py fix uses cfg=1.0 for both modes (upstream had cfg=0 for turbo).
        cfg_scale = 1.0

        def model_fn(x, t, **_):
            tt = (torch.tensor([t], device=x.device, dtype=torch.float32)
                  if not isinstance(t, torch.Tensor) else t.float())
            if tt.dim() == 0:
                tt = tt.unsqueeze(0)
            if tt.shape[0] == 1 and x.shape[0] > 1:
                tt = tt.expand(x.shape[0])
            if cfg_scale > 0:
                out = diff_model(
                    x=list(x.to(torch.bfloat16).repeat(2, 1, 1, 1, 1).unbind(0)),
                    t=tt.repeat(2), cap_feats=doubled,
                    patch_size=patch, f_patch_size=fpatch, return_dict=False,
                )
                pos, neg = out[0][:n], out[0][n:]
                res = []
                for p, ng in zip(pos, neg):
                    p, ng = p.float(), ng.float()
                    pred = p + cfg_scale * (p - ng)
                    on, nn_ = torch.linalg.vector_norm(p), torch.linalg.vector_norm(pred)
                    if nn_ > on:
                        pred *= on / nn_
                    res.append(pred)
                return torch.stack(res)
            out = diff_model(x=list(x.to(torch.bfloat16).unbind(0)), t=tt,
                             cap_feats=cap_pos, patch_size=patch,
                             f_patch_size=fpatch, return_dict=False)
            return torch.stack([o.float() for o in out[0]])

        sampler = Sampler(create_transport("Linear", "velocity", None))
        sample_fn = sampler.sample_ode(
            sampling_method="euler", num_steps=num_steps,
            atol=1e-6, rtol=1e-3, reverse=False, time_shifting_factor=6,
            stochast_ratio=0.0,
        )

        step_i = [0]

        def progress_wrap(x, t, **kw):
            step_i[0] += 1
            if progress_cb is not None:
                progress_cb(step_i[0] / max(num_steps, 1), f"decode {step_i[0]}/{num_steps}")
            return model_fn(x, t, **kw)

        samples = sample_fn(z, progress_wrap)[-1].squeeze(2)

        # Stage 3: VAE decode
        s = samples.to(torch.bfloat16)
        s = (s / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        px = ((self.vae.decode(s, return_dict=False)[0] + 1) / 2).clamp_(0, 1)
        return to_pil_image(px[0].float())

    # --- high-level tasks ---
    def t2i(
        self, prompt: str, image_h: int, image_w: int, llm_steps: int,
        cfg_scale: float, decoder_mode: str, decoder_steps: int | None,
        resolution_multiplier: int, seed: int, progress_cb=None,
    ) -> tuple[Image.Image, dict, float, float]:
        # Free the decoder before bringing the LLM up (low_vram / lod).
        if self.low_vram or self.lod:
            self.unload_decoder()
        self.load_llm()
        torch.manual_seed(seed)
        t0 = time.time()
        if progress_cb:
            progress_cb(0.0, "LLM: generating VQ tokens …")
        # generate_image's default gen_length=1088 only fits a 32x32 (=1024 tokens)
        # grid plus a small header. Anything bigger silently truncates the
        # returned token_ids, which then fails the .view(1, 1, h, w) reshape in
        # the decoder. Auto-size from the requested dims (model halves them
        # internally, then /16 to get the token grid) and add a small header
        # safety margin.
        grid_h = (image_h // 2) // 16
        grid_w = (image_w // 2) // 16
        needed = grid_h * grid_w + 64
        gen_length = max(1088, needed)
        res = self.llm.generate_image(
            prompt, image_h=image_h, image_w=image_w,
            steps=llm_steps, cfg_scale=cfg_scale,
            gen_length=gen_length,
        )
        t_llm = time.time() - t0

        meta = {
            "task": "t2i", "prompt": prompt, "seed": seed,
            "image_h": image_h, "image_w": image_w,
            "llm_steps": llm_steps, "cfg_scale": cfg_scale,
            "quant": self.quant, "model_path": self.model_path,
        }
        self.last_vq = {
            "token_ids": res["token_ids"], "h": res["h"], "w": res["w"], "meta": meta,
        }

        if progress_cb:
            progress_cb(0.0, "Decoder: running …")
        t1 = time.time()
        img = self.decode(
            res["token_ids"], res["h"], res["w"],
            decoder_mode=decoder_mode, num_steps=decoder_steps,
            resolution_multiplier=resolution_multiplier, seed=seed,
            progress_cb=progress_cb,
        )
        t_dec = time.time() - t1
        return img, meta, t_llm, t_dec

    def mmu(
        self, image_path: str, question: str, steps: int, block_length: int,
        gen_length: int, seed: int,
    ) -> tuple[str, float]:
        from decoder.utils import generate_crop_size_list, var_center_crop

        if self.low_vram or self.lod:
            self.unload_decoder()
        self.load_llm()
        self.load_image_tokenizer()
        torch.manual_seed(seed)
        offset = _image_token_offset(self.model_path)

        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        pil = var_center_crop(
            Image.open(image_path).convert("RGB"), crop_size_list=crop_size_list,
        )
        info = self.image_tokenizer.encode_with_info(pil)
        _, h, w = info["grid_thw"]
        ids = [x + offset for x in info["token_ids"]]

        t0 = time.time()
        ans = self.llm.understand_image(
            ids, h, w, question=question,
            steps=steps, block_length=block_length, gen_length=gen_length,
        )
        return ans, time.time() - t0

    def edit(
        self, image_path: str, instruction: str, input_size: int, steps: int,
        block_length: int, cfg_text_scale: float, cfg_image_scale: float,
        decoder_mode: str, decoder_steps: int | None,
        resolution_multiplier: int, seed: int, progress_cb=None,
    ) -> tuple[Image.Image, dict, float, float]:
        from decoder.utils import generate_crop_size_list, var_center_crop

        if self.low_vram or self.lod:
            self.unload_decoder()
        self.load_llm()
        self.load_image_tokenizer()
        torch.manual_seed(seed)
        offset = _image_token_offset(self.model_path)

        if progress_cb:
            progress_cb(0.0, "Tokenizing input image …")
        crop_size_list = generate_crop_size_list((input_size // 32) ** 2, 32)
        pil = var_center_crop(
            Image.open(image_path).convert("RGB"), crop_size_list=crop_size_list,
        )
        info = self.image_tokenizer.encode_with_info(pil)
        _, h, w = info["grid_thw"]
        ids = [x + offset for x in info["token_ids"]]

        if progress_cb:
            progress_cb(0.0, "LLM: applying edit …")
        t0 = time.time()
        res = self.llm.edit_image(
            ids, h, w, instruction,
            steps=steps, block_length=block_length,
            cfg_text_scale=cfg_text_scale, cfg_image_scale=cfg_image_scale,
        )
        t_llm = time.time() - t0

        meta = {
            "task": "edit", "image": str(image_path), "instruction": instruction,
            "seed": seed, "steps": steps, "block_length": block_length,
            "cfg_text_scale": cfg_text_scale, "cfg_image_scale": cfg_image_scale,
            "input_size": input_size, "quant": self.quant,
            "model_path": self.model_path,
        }
        self.last_vq = {
            "token_ids": res["token_ids"], "h": res["h"], "w": res["w"], "meta": meta,
        }

        if progress_cb:
            progress_cb(0.0, "Decoder: running …")
        t1 = time.time()
        img = self.decode(
            res["token_ids"], res["h"], res["w"],
            decoder_mode=decoder_mode, num_steps=decoder_steps,
            resolution_multiplier=resolution_multiplier, seed=seed,
            progress_cb=progress_cb,
        )
        t_dec = time.time() - t1
        return img, meta, t_llm, t_dec


# ---------------------------------------------------------------------------
# Gradio wiring
# ---------------------------------------------------------------------------
def build_app(pipe: Pipeline):
    import gradio as gr

    with gr.Blocks(title="LLaDA-2.0-Uni") as app:
        header = (
            f"# LLaDA-2.0-Uni · Unified dLLM\n"
            f"Backbone: **{pipe.quant}** · Path: `{pipe.model_path}` · "
            f"Device: **{pipe.device}**"
        )
        if pipe.low_vram:
            header += (
                "\n\n*Running in `--low_vram` mode: LLM and decoder swap on and "
                "off GPU, adding ~75 s of LLM reload between generations. Use "
                "the **Replay decoder** tab to iterate on decoder settings "
                "without the reload cost.*"
            )
        gr.Markdown(header)

        # ---- T2I ----
        with gr.Tab("Text → Image"):
            with gr.Row():
                with gr.Column(scale=2):
                    t2i_prompt = gr.Textbox(
                        label="Prompt", lines=3,
                        placeholder="a red panda wearing sunglasses sitting on a bamboo chair, photo-realistic",
                    )
                    with gr.Row():
                        t2i_h = gr.Slider(256, 2048, 1024, step=64, label="Image H (px)")
                        t2i_w = gr.Slider(256, 2048, 1024, step=64, label="Image W (px)")
                    with gr.Row():
                        t2i_llm_steps = gr.Slider(2, 32, 8, step=1, label="LLM steps")
                        t2i_cfg = gr.Slider(0.0, 10.0, 2.0, step=0.1, label="LLM CFG scale")
                    with gr.Row():
                        t2i_mode = gr.Radio(
                            ["turbo", "normal"], value="turbo",
                            label="Decoder",
                        )
                        t2i_dec_steps = gr.Slider(
                            2, 50, 8, step=1,
                            label="Decoder steps (auto if default)",
                        )
                    with gr.Row():
                        t2i_seed = gr.Number(value=42, label="Seed", precision=0)
                    t2i_go = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    t2i_img = gr.Image(label="Output", type="pil", height=600)
                    t2i_info = gr.Markdown("", elem_classes="status-box")

            def run_t2i(prompt, h, w, ls, cfg, mode, ds, seed,
                        progress=gr.Progress()):
                if not prompt.strip():
                    raise gr.Error("Prompt is empty")

                def cb(frac, msg):
                    progress(frac, desc=msg)

                try:
                    progress(0.0, desc="LLM warming up …")
                    img, meta, t_llm, t_dec = pipe.t2i(
                        prompt, int(h), int(w), int(ls), float(cfg),
                        mode, int(ds), _FIXED_RES_MULT, int(seed), cb,
                    )
                    info = (f"LLM: {t_llm:.1f}s · Decoder: {t_dec:.1f}s · "
                            f"Total: {t_llm + t_dec:.1f}s\n\n"
                            f"Tokens: {meta['image_h'] // 32} × {meta['image_w'] // 32} = "
                            f"{(meta['image_h'] // 32) * (meta['image_w'] // 32)} "
                            f"· CUDA: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
                    return img, info
                finally:
                    pipe.finalize_request()

            t2i_go.click(
                run_t2i,
                [t2i_prompt, t2i_h, t2i_w, t2i_llm_steps, t2i_cfg,
                 t2i_mode, t2i_dec_steps, t2i_seed],
                [t2i_img, t2i_info],
            )

        # ---- MMU ----
        with gr.Tab("Understand"):
            with gr.Row():
                with gr.Column(scale=2):
                    mmu_img = gr.Image(type="filepath", label="Input image", height=400)
                    mmu_q = gr.Textbox(
                        label="Question", lines=2,
                        value="Describe this image in detail.",
                    )
                    with gr.Row():
                        mmu_steps = gr.Slider(2, 64, 32, step=1, label="LLM steps")
                        mmu_blk = gr.Slider(16, 64, 32, step=4, label="Block length")
                        mmu_len = gr.Slider(32, 1024, 256, step=32, label="Gen length")
                    mmu_seed = gr.Number(value=42, label="Seed", precision=0)
                    mmu_go = gr.Button("Ask", variant="primary")
                with gr.Column(scale=3):
                    mmu_ans = gr.Textbox(label="Answer", lines=14)
                    mmu_info = gr.Markdown("", elem_classes="status-box")

            def run_mmu(path, q, s, bl, gl, seed):
                if not path:
                    raise gr.Error("Upload an image first.")
                try:
                    ans, t = pipe.mmu(path, q, int(s), int(bl), int(gl), int(seed))
                    return ans, f"LLM: {t:.1f}s"
                finally:
                    pipe.finalize_request()

            mmu_go.click(run_mmu,
                         [mmu_img, mmu_q, mmu_steps, mmu_blk, mmu_len, mmu_seed],
                         [mmu_ans, mmu_info])

        # ---- EDIT ----
        with gr.Tab("Edit"):
            with gr.Row():
                with gr.Column(scale=2):
                    ed_img = gr.Image(type="filepath", label="Source image", height=400)
                    ed_instr = gr.Textbox(
                        label="Edit instruction", lines=2,
                        placeholder="replace the window view with snowy mountains",
                    )
                    with gr.Row():
                        ed_input = gr.Slider(512, 1536, 1024, step=128,
                                             label="Input size (px)")
                        ed_steps = gr.Slider(2, 32, 8, step=1, label="LLM steps")
                        ed_blk = gr.Slider(16, 64, 32, step=4, label="Block length")
                    with gr.Row():
                        ed_cfg_t = gr.Slider(0.0, 10.0, 4.0, step=0.1, label="CFG text")
                        ed_cfg_i = gr.Slider(0.0, 10.0, 0.0, step=0.1, label="CFG image")
                    with gr.Row():
                        ed_mode = gr.Radio(["turbo", "normal"], value="turbo",
                                           label="Decoder")
                        ed_ds = gr.Slider(2, 50, 8, step=1, label="Decoder steps")
                    ed_seed = gr.Number(value=42, label="Seed", precision=0)
                    ed_go = gr.Button("Edit", variant="primary")
                with gr.Column(scale=3):
                    ed_out = gr.Image(label="Output", type="pil", height=500)
                    ed_info = gr.Markdown("", elem_classes="status-box")

            def run_edit(path, instr, isz, stp, blk, ct, ci, mode, ds, seed,
                         progress=gr.Progress()):
                if not path:
                    raise gr.Error("Upload a source image first.")
                if not instr.strip():
                    raise gr.Error("Instruction is empty.")

                def cb(frac, msg):
                    progress(frac, desc=msg)

                try:
                    progress(0.0, desc="Tokenizing …")
                    img, meta, t_llm, t_dec = pipe.edit(
                        path, instr, int(isz), int(stp), int(blk),
                        float(ct), float(ci), mode, int(ds),
                        _FIXED_RES_MULT, int(seed), cb,
                    )
                    info = (f"LLM: {t_llm:.1f}s · Decoder: {t_dec:.1f}s · "
                            f"Total: {t_llm + t_dec:.1f}s\n\n"
                            f"CUDA: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
                    return img, info
                finally:
                    pipe.finalize_request()

            ed_go.click(
                run_edit,
                [ed_img, ed_instr, ed_input, ed_steps, ed_blk, ed_cfg_t, ed_cfg_i,
                 ed_mode, ed_ds, ed_seed],
                [ed_out, ed_info],
            )

        # ---- Replay decoder ----
        with gr.Tab("Replay decoder"):
            gr.Markdown(
                "Re-runs the decoder on the **last VQ tokens** from the current "
                "session (whichever t2i or edit was most recent) without re-"
                "running the LLM. Useful for comparing decoder settings side-by-side."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    rep_mode = gr.Radio(["turbo", "normal"], value="turbo",
                                        label="Decoder")
                    rep_ds = gr.Slider(2, 50, 8, step=1, label="Decoder steps")
                    rep_seed = gr.Number(value=42, label="Seed", precision=0)
                    rep_go = gr.Button("Re-decode", variant="primary")
                    gr.Markdown("Or upload a saved `.pt` tokens file:")
                    rep_upload = gr.File(label="VQ tokens (.pt)", file_types=[".pt"])
                with gr.Column(scale=3):
                    rep_out = gr.Image(label="Output", type="pil", height=500)
                    rep_info = gr.Markdown("", elem_classes="status-box")

            def run_replay(mode, ds, seed, upload,
                           progress=gr.Progress()):
                if upload is not None:
                    payload = torch.load(upload, map_location="cpu",
                                         weights_only=False)
                    tok_ids = payload["token_ids"]
                    h, w = payload["h"], payload["w"]
                    meta = payload.get("meta", {})
                elif pipe.last_vq is not None:
                    tok_ids = pipe.last_vq["token_ids"]
                    h, w = pipe.last_vq["h"], pipe.last_vq["w"]
                    meta = pipe.last_vq["meta"]
                else:
                    raise gr.Error(
                        "No cached VQ tokens yet — generate or edit once "
                        "first, or upload a .pt tokens file."
                    )

                def cb(frac, msg):
                    progress(frac, desc=msg)

                try:
                    t0 = time.time()
                    img = pipe.decode(
                        tok_ids, h, w, decoder_mode=mode,
                        num_steps=int(ds), resolution_multiplier=_FIXED_RES_MULT,
                        seed=int(seed), progress_cb=cb,
                    )
                    dt = time.time() - t0
                    info = (
                        f"Decoder: {dt:.1f}s · grid {h}×{w} · "
                        f"source task: {meta.get('task','?')} · "
                        f"'{(meta.get('prompt') or meta.get('instruction') or '')[:80]}'"
                    )
                    return img, info
                finally:
                    pipe.finalize_request()

            rep_go.click(
                run_replay,
                [rep_mode, rep_ds, rep_seed, rep_upload],
                [rep_out, rep_info],
            )

        # ---- Status / Info ----
        with gr.Tab("Status"):
            stat = gr.Markdown()

            def refresh():
                lines = [
                    f"**Quant:** `{pipe.quant}`",
                    f"**Model path:** `{pipe.model_path}`",
                    f"**CUDA mem:** {torch.cuda.memory_allocated() / 1e9:.2f} GB",
                    f"**low_vram:** {'on (swapping enabled)' if pipe.low_vram else 'off'}",
                    f"**lod:** {'on (cold-start every request)' if pipe.lod else 'off'}",
                    "",
                    "### Loaded components",
                    f"- LLM: {'✓ loaded' if pipe.llm is not None else '— not loaded'}",
                    f"- SigVQ: {'✓ loaded' if pipe.sigvq is not None else '— not loaded'}",
                    f"- Decoder modes loaded: "
                    f"{', '.join(pipe.decoder_models) or '—'}",
                    f"- VAE: {'✓ loaded' if pipe.vae is not None else '— not loaded'}",
                    f"- Image tokenizer: "
                    f"{'✓ loaded' if pipe.image_tokenizer is not None else '— not loaded'}",
                    "",
                    f"**Last VQ tokens cached:** "
                    f"{'yes (' + pipe.last_vq['meta'].get('task','?') + ')' if pipe.last_vq else 'no'}",
                ]
                return "\n".join(lines)

            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(refresh, None, stat)
            app.load(refresh, None, stat)

    return app


# ---------------------------------------------------------------------------
# FastAPI HTTP API (for programmatic / ComfyUI access)
# ---------------------------------------------------------------------------
def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image → base64 string, never touching the disk."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@contextmanager
def _b64_to_tempfile(b64: str, suffix: str = ".png"):
    """Write a base64-decoded image to a temp file, yield path, cleanup on exit.

    Needed because Pipeline.edit / Pipeline.mmu call ``Image.open(path)``
    inside ``var_center_crop`` — refactoring them to accept bytes or a PIL
    image would touch model-internal code paths, temp files are safer.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(b64))
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build_api(pipe: "Pipeline"):
    """Construct a FastAPI app exposing t2i / edit / mmu / decode / status.

    All image I/O is in-memory base64. Nothing touches the disk beyond a
    transient tempfile for the input side of edit / mmu (deleted before
    the response is returned).
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class T2IRequest(BaseModel):
        prompt: str
        image_h: int = 1024
        image_w: int = 1024
        llm_steps: int = 8
        cfg_scale: float = 2.0
        decoder_mode: str = Field("turbo", pattern="^(turbo|normal)$")
        decoder_steps: Optional[int] = None
        seed: int = 42

    class EditRequest(BaseModel):
        image_base64: str
        instruction: str
        input_size: int = 1024
        llm_steps: int = 8
        block_length: int = 32
        cfg_text_scale: float = 4.0
        cfg_image_scale: float = 0.0
        decoder_mode: str = Field("turbo", pattern="^(turbo|normal)$")
        decoder_steps: Optional[int] = None
        seed: int = 42

    class MMURequest(BaseModel):
        image_base64: str
        question: str = "Describe this image in detail."
        llm_steps: int = 32
        block_length: int = 32
        gen_length: int = 256
        seed: int = 42

    api = FastAPI(
        title="LLaDA-2.0-Uni API",
        version="0.1.0",
        description=(
            "HTTP API around the LLaDA-2.0-Uni unified dLLM. All responses "
            "return base64-encoded PNG data inline; nothing is written to "
            "disk by the server. Pair with ``--lod`` to idle at 0 GB VRAM "
            "between calls."
        ),
    )

    @api.get("/api/status")
    def status():
        return {
            "quant": pipe.quant,
            "model_path": pipe.model_path,
            "device": pipe.device,
            "low_vram": pipe.low_vram,
            "lod": pipe.lod,
            "cuda_allocated_gb": round(
                torch.cuda.memory_allocated() / 1e9, 3,
            ),
            "loaded": {
                "llm": pipe.llm is not None,
                "sigvq": pipe.sigvq is not None,
                "vae": pipe.vae is not None,
                "image_tokenizer": pipe.image_tokenizer is not None,
                "decoders": list(pipe.decoder_models.keys()),
            },
            "last_vq_cached": bool(pipe.last_vq),
        }

    @api.post("/api/t2i")
    def t2i(req: T2IRequest):
        try:
            img, meta, t_llm, t_dec = pipe.t2i(
                req.prompt, req.image_h, req.image_w, req.llm_steps,
                req.cfg_scale, req.decoder_mode, req.decoder_steps,
                _FIXED_RES_MULT, req.seed,
            )
        except Exception as e:
            pipe.finalize_request()
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        pipe.finalize_request()
        return {
            "image_base64": _pil_to_b64(img),
            "format": "png",
            "meta": meta,
            "timing": {"t_llm": t_llm, "t_dec": t_dec, "total": t_llm + t_dec},
        }

    @api.post("/api/edit")
    def edit(req: EditRequest):
        try:
            with _b64_to_tempfile(req.image_base64) as path:
                img, meta, t_llm, t_dec = pipe.edit(
                    path, req.instruction, req.input_size, req.llm_steps,
                    req.block_length, req.cfg_text_scale, req.cfg_image_scale,
                    req.decoder_mode, req.decoder_steps,
                    _FIXED_RES_MULT, req.seed,
                )
        except Exception as e:
            pipe.finalize_request()
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        pipe.finalize_request()
        # Strip the tempfile path out of meta — it's dead after the request.
        meta = {k: v for k, v in meta.items() if k != "image"}
        return {
            "image_base64": _pil_to_b64(img),
            "format": "png",
            "meta": meta,
            "timing": {"t_llm": t_llm, "t_dec": t_dec, "total": t_llm + t_dec},
        }

    @api.post("/api/mmu")
    def mmu(req: MMURequest):
        try:
            with _b64_to_tempfile(req.image_base64) as path:
                answer, t_llm = pipe.mmu(
                    path, req.question, req.llm_steps, req.block_length,
                    req.gen_length, req.seed,
                )
        except Exception as e:
            pipe.finalize_request()
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        pipe.finalize_request()
        return {
            "answer": answer,
            "timing": {"t_llm": t_llm, "total": t_llm},
        }

    return api


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["nf4", "fp8", "bf16"], default="nf4",
                    help="Backbone precision (default: nf4 — 9.5 GB VRAM)")
    ap.add_argument("--model_path", default=None,
                    help="Override auto-selected dir for --quant")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Expose a public gradio.live URL (UI only, no effect in --api).")
    ap.add_argument("--no-eager-load", action="store_true",
                    help="Skip loading the LLM at launch (lazy on first request). "
                         "Implied by --lod.")
    ap.add_argument("--low_vram", action="store_true",
                    help="Swap LLM and decoder on and off GPU so they never "
                         "co-reside. Required for 24 GB cards (4090/3090). "
                         "Adds ~75 s LLM reload between generations.")
    ap.add_argument("--lod", action="store_true",
                    help="Load-on-demand: unload EVERYTHING from GPU after each "
                         "request so the process idles at ~0 GB VRAM. Every call "
                         "pays the full reload cost (~90 s cold). Great for "
                         "long-running servers that don't want to hog the GPU.")
    ap.add_argument("--api", action="store_true",
                    help="Enable the HTTP API at /api/* (FastAPI). Compatible "
                         "with --lod. See /docs on the running server for the "
                         "OpenAPI schema. Images are returned as base64 PNG "
                         "inline and never saved to disk.")
    args = ap.parse_args()

    model_path = _ensure_model_path(args.quant, args.model_path)
    pipe = Pipeline(
        model_path=model_path, quant=args.quant, device=args.device,
        low_vram=args.low_vram, lod=args.lod,
    )
    if args.low_vram:
        print("[low_vram] enabled — LLM ↔ decoder will be swapped between stages")
    if args.lod:
        print("[lod] enabled — all components unload after each request "
              "(process idles at 0 GB VRAM)")

    # --lod implies no eager load: pointless to load the LLM only to tear it
    # right back down the first time finalize_request runs.
    eager = (not args.no_eager_load) and (not args.lod)
    if eager:
        print("Eager-loading LLM backbone at startup; pass --no-eager-load to skip.")
        pipe.load_llm()
    else:
        print("Skipping eager load; first request will pay the ~75 s LLM load.")

    import gradio as gr

    gradio_app = build_app(pipe)

    if args.api:
        # Mount the Gradio UI inside a FastAPI app so both the /ui and the
        # /api/* endpoints share a single uvicorn server and a single pipe.
        import uvicorn
        from fastapi.responses import RedirectResponse

        api = build_api(pipe)

        # Redirect root to the UI so bare http://host:port/ still gives humans
        # something useful while /api/* serves machines and /docs serves the
        # OpenAPI explorer.
        @api.get("/", include_in_schema=False)
        def _root():
            return RedirectResponse(url="/ui")

        api = gr.mount_gradio_app(api, gradio_app, path="/ui")
        print(f"[api] UI:   http://{args.host}:{args.port}/ui")
        print(f"[api] API:  http://{args.host}:{args.port}/api/t2i (see /docs)")
        uvicorn.run(api, host=args.host, port=args.port, log_level="info")
    else:
        gradio_app.launch(
            server_name=args.host, server_port=args.port, share=args.share,
            show_error=True, theme=gr.themes.Soft(),
        )


if __name__ == "__main__":
    main()
