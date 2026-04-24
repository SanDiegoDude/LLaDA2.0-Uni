"""
LLaDA-2.0-Uni — unified inference CLI.

Subcommands:
    t2i   PROMPT                        text-to-image
    mmu   IMAGE [--question STR]        image understanding
    edit  IMAGE INSTRUCTION             image editing
    info                                print detected model variants

Backbone precision is chosen with ``--quant``:
    nf4   (default) — bnb NF4, pre-quantized on disk, ~9.5 GB VRAM, ~70 s load
    fp8             — torchao FP8 (inline quant from bf16), ~17 GB VRAM, ~200 s load
    bf16            — full bf16, ~32 GB VRAM, fastest inference

Decoder defaults to ``turbo`` — the distilled 8-step decoder, ~6× faster than
``normal`` (8 vs 50 ODE steps) with equivalent quality once the upstream
``cfg_scale=0.0``/``stochast_ratio=1.0`` turbo defaults are corrected to match
the normal decoder (cfg=1.0, pure ODE). See ``decoder/decode.py`` for the fix.
Pass ``--decoder_mode normal`` for the 50-step path if ever needed.

The --model_path directory must contain:
    - LLM backbone weights + tokenizer  (for nf4: the pre-quantized artifact)
    - image_tokenizer/, decoder/, decoder-turbo/, vae/  (bf16 aux components)
By default --model_path is auto-picked from --quant; override if needed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)

# Repo root on sys.path so decoder / encoder packages import
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Backbone locations — NF4 is self-contained (has aux dirs copied in), bf16 source
# is used as base for either bf16 inference or inline FP8 quant at load time.
MODEL_PATHS = {
    "nf4": "/home/nathan/shared/models/LLaDA2.0-Uni-nf4",
    "bf16": "/home/nathan/shared/models/LLaDA2.0-Uni",
    "fp8": "/home/nathan/shared/models/LLaDA2.0-Uni",  # inline-quanted from bf16
}
DEFAULT_QUANT = "nf4"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _slugify(s: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_]+", "_", s)
    return s[:maxlen] or "output"


def _auto_output_path(mode: str, seed: int, stub: str, ext: str = "png") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(stub)
    return f"./outputs/{mode}_{ts}_seed{seed}_{slug}.{ext}"


def _is_quantized(model_path: str) -> bool:
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        return "quantization_config" in json.load(f)


def _image_token_offset(model_path: str) -> int:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f).get("image_token_offset", 157184)


def _build_fp8_config():
    """torchao FP8 config that actually runs on GB10 / sm_121.

    Per-row scales hit an sm_90a-gated MMA kernel that aborts on Blackwell, so
    we're stuck with per-tensor. Dynamic per-token activation scaling rescues
    quality. KernelPreference.TORCH bypasses the buggy cuBLAS auto-select.
    """
    from transformers import TorchAoConfig
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerTensor,
    )
    from torchao.quantization.quantize_.common.kernel_preference import (
        KernelPreference,
    )

    cfg = Float8DynamicActivationFloat8WeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        granularity=PerTensor(),
        kernel_preference=KernelPreference.TORCH,
    )
    return TorchAoConfig(
        quant_type=cfg,
        modules_to_not_convert=["lm_head", "gate"],
    )


def _load_llm(model_path: str, device: str, quant: str):
    """Quant-aware LLM load.

    - ``nf4``: loads a pre-quantized bnb artifact from disk. Fast (~70s).
    - ``bf16``: loads bf16 weights directly. Faster (~60s) but 32 GB VRAM.
    - ``fp8``: loads bf16 weights and inline-quantizes with torchao. Slower
      (~200s) because every layer is re-scaled on the way in, but produces no
      on-disk artifact and avoids the transformers-4.57 ↔ torchao-0.14
      safetensors sub-key incompatibility.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_stored_quant = _is_quantized(model_path)
    print(f"  [LLM] Loading {quant} backbone from {model_path}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs = dict(
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
    )
    if quant == "fp8":
        load_kwargs["quantization_config"] = _build_fp8_config()

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()

    # Pre-quantized (nf4) and torchao fp8 models already carry their own dtype
    # scheme; only full-precision bf16 needs an explicit cast.
    if quant == "bf16" and not is_stored_quant:
        model = model.to(torch.bfloat16)

    model.tokenizer = tokenizer
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  [LLM] loaded in {time.time() - t0:.1f}s, CUDA mem: {mem:.2f} GB")
    return model


def _unload(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def _encode_image(image_path: str, model_path: str, device: str, offset: int,
                  target_size: int = 1024):
    from PIL import Image

    from encoder.image_tokenizer import ImageTokenizer
    from decoder.utils import generate_crop_size_list, var_center_crop

    print(f"  [IMG-ENC] Loading image tokenizer for {image_path}")
    t0 = time.time()
    itok = ImageTokenizer(model_path=model_path, device=device, dtype=torch.bfloat16)
    crop_size_list = generate_crop_size_list((target_size // 32) ** 2, 32)
    pil = var_center_crop(
        Image.open(image_path).convert("RGB"), crop_size_list=crop_size_list
    )
    info = itok.encode_with_info(pil)
    _, h, w = info["grid_thw"]
    ids = [x + offset for x in info["token_ids"]]
    _unload(itok)
    print(f"  [IMG-ENC] done in {time.time() - t0:.1f}s — grid {h}x{w}, {len(ids)} tokens")
    return ids, h, w


def _save_vq_tokens(path: str, res: dict, meta: dict) -> None:
    """Dump LLM VQ output + metadata to a .pt file so we can iterate on the
    decoder without re-running the 8-minute LLM pass."""
    payload = {
        "token_ids": res["token_ids"],
        "h": res["h"],
        "w": res["w"],
        "meta": meta,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"  [VQ] saved tokens + meta to {path}")


# ---------------------------------------------------------------------------
# t2i
# ---------------------------------------------------------------------------
def cmd_t2i(args: argparse.Namespace) -> None:
    from decoder import decode_vq_tokens

    torch.manual_seed(args.seed)
    device = args.device

    out_path = args.output or _auto_output_path("t2i", args.seed, args.prompt)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== T2I: {args.prompt[:80]} ===")
    total_t = time.time()

    # Stage 1: LLM → VQ tokens
    print("\n[1/3] LLM text→VQ generation")
    t0 = time.time()
    model = _load_llm(args.model_path, device, args.quant)
    res = model.generate_image(
        args.prompt,
        image_h=args.image_h,
        image_w=args.image_w,
        steps=args.llm_steps,
        cfg_scale=args.cfg_scale,
    )
    print(f"  [LLM] generated {len(res['token_ids'])} VQ tokens in {time.time() - t0:.1f}s")
    _unload(model)

    if args.save_vq_tokens:
        _save_vq_tokens(args.save_vq_tokens, res,
                        {"task": "t2i", "prompt": args.prompt, "seed": args.seed,
                         "image_h": args.image_h, "image_w": args.image_w,
                         "llm_steps": args.llm_steps, "cfg_scale": args.cfg_scale,
                         "quant": args.quant, "model_path": args.model_path})

    # Stage 2+3: SigVQ + diffusion decode + VAE  (handled by decode_vq_tokens)
    dec_steps = args.decoder_steps or (8 if args.decoder_mode == "turbo" else 50)
    print(f"\n[2/3] Diffusion decode ({args.decoder_mode}, {dec_steps} steps)")
    t0 = time.time()
    img = decode_vq_tokens(
        res["token_ids"],
        res["h"],
        res["w"],
        args.model_path,
        device,
        resolution_multiplier=args.resolution_multiplier,
        num_steps=dec_steps,
        decode_mode="decoder-turbo" if args.decoder_mode == "turbo" else "normal",
    )
    print(f"  [DEC] decoded in {time.time() - t0:.1f}s")

    print(f"\n[3/3] Saving → {out_path}")
    img.save(out_path)
    print(f"\nDone ({time.time() - total_t:.1f}s total)")


# ---------------------------------------------------------------------------
# mmu
# ---------------------------------------------------------------------------
def cmd_mmu(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = args.device

    print(f"\n=== MMU: {args.image}  Q: {args.question[:80]} ===")
    total_t = time.time()

    offset = _image_token_offset(args.model_path)
    image_tokens, h, w = _encode_image(args.image, args.model_path, device, offset)

    print("\n[LLM] Loading backbone + running understanding")
    model = _load_llm(args.model_path, device, args.quant)
    t0 = time.time()
    answer = model.understand_image(
        image_tokens,
        h,
        w,
        question=args.question,
        steps=args.steps,
        block_length=args.block_length,
        gen_length=args.gen_length,
    )
    print(f"  [LLM] answered in {time.time() - t0:.1f}s")
    _unload(model)

    print(f"\n{'=' * 60}\nQ: {args.question}\n{'-' * 60}\n{answer}\n{'=' * 60}")
    print(f"\nDone ({time.time() - total_t:.1f}s total)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(f"Image: {args.image}\nQuestion: {args.question}\n\n{answer}\n")
        print(f"Answer written to {args.output}")


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------
def cmd_edit(args: argparse.Namespace) -> None:
    from decoder import decode_vq_tokens

    torch.manual_seed(args.seed)
    device = args.device

    out_path = args.output or _auto_output_path("edit", args.seed, args.instruction)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== EDIT: {args.image}")
    print(f"    instruction: {args.instruction[:80]}")
    total_t = time.time()

    offset = _image_token_offset(args.model_path)
    image_tokens, h, w = _encode_image(args.image, args.model_path, device, offset,
                                       target_size=args.input_size)

    print("\n[1/2] LLM edit→VQ")
    t0 = time.time()
    model = _load_llm(args.model_path, device, args.quant)
    res = model.edit_image(
        image_tokens, h, w, args.instruction,
        steps=args.steps,
        block_length=args.block_length,
        cfg_text_scale=args.cfg_text_scale,
        cfg_image_scale=args.cfg_image_scale,
    )
    print(f"  [LLM] edited in {time.time() - t0:.1f}s — {len(res['token_ids'])} VQ tokens")
    _unload(model)

    if args.save_vq_tokens:
        _save_vq_tokens(args.save_vq_tokens, res,
                        {"task": "edit", "image": args.image,
                         "instruction": args.instruction, "seed": args.seed,
                         "steps": args.steps, "block_length": args.block_length,
                         "cfg_text_scale": args.cfg_text_scale,
                         "cfg_image_scale": args.cfg_image_scale,
                         "quant": args.quant, "model_path": args.model_path})

    dec_steps = args.decoder_steps or (8 if args.decoder_mode == "turbo" else 50)
    print(f"\n[2/2] Diffusion decode ({args.decoder_mode}, {dec_steps} steps)")
    t0 = time.time()
    img = decode_vq_tokens(
        res["token_ids"], res["h"], res["w"],
        args.model_path, device,
        resolution_multiplier=args.resolution_multiplier,
        num_steps=dec_steps,
        decode_mode="decoder-turbo" if args.decoder_mode == "turbo" else "normal",
    )
    print(f"  [DEC] decoded in {time.time() - t0:.1f}s")
    img.save(out_path)
    print(f"\nSaved → {out_path}")
    print(f"Done ({time.time() - total_t:.1f}s total)")


# ---------------------------------------------------------------------------
# decode (replay decoder on saved VQ tokens)
# ---------------------------------------------------------------------------
def cmd_decode(args: argparse.Namespace) -> None:
    from decoder import decode_vq_tokens

    torch.manual_seed(args.seed)
    device = args.device

    payload = torch.load(args.tokens_file, map_location="cpu", weights_only=False)
    token_ids = payload["token_ids"]
    h, w = payload["h"], payload["w"]
    meta = payload.get("meta", {})

    stub = meta.get("prompt") or meta.get("instruction") or "decode"
    out_path = args.output or _auto_output_path("decode", args.seed, stub)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== DECODE from {args.tokens_file} ===")
    if meta:
        print(f"  [META] task={meta.get('task')} "
              f"prompt/instruction='{(meta.get('prompt') or meta.get('instruction') or '')[:60]}'")
    print(f"  [VQ] {len(token_ids)} tokens, grid {h}x{w}")

    dec_steps = args.decoder_steps or (8 if args.decoder_mode == "turbo" else 50)
    print(f"\n[DEC] Diffusion decode ({args.decoder_mode}, {dec_steps} steps)")
    t0 = time.time()
    img = decode_vq_tokens(
        token_ids, h, w,
        args.model_path, device,
        resolution_multiplier=args.resolution_multiplier,
        num_steps=dec_steps,
        decode_mode="decoder-turbo" if args.decoder_mode == "turbo" else "normal",
    )
    print(f"  [DEC] decoded in {time.time() - t0:.1f}s")
    img.save(out_path)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> None:
    p = args.model_path
    print(f"Requested quant: {args.quant}")
    print(f"Model path:      {p}")
    if not os.path.isdir(p):
        print("  (directory does not exist)")
        return
    cfg_path = os.path.join(p, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        qcfg = cfg.get("quantization_config")
        if qcfg:
            print(f"  LLM on disk: QUANTIZED ({qcfg.get('quant_method', '?')})")
            for k in ("load_in_4bit", "load_in_8bit", "bnb_4bit_quant_type", "quant_type"):
                if k in qcfg and qcfg[k]:
                    print(f"    {k}: {qcfg[k]}")
        else:
            print("  LLM on disk: bf16 (no quantization_config in config.json)")
            if args.quant == "fp8":
                print("  At load time: torchao FP8 (dyn-act + per-tensor weight) will be "
                      "applied inline — expect ~200s extra on first use.")
            elif args.quant == "nf4":
                print("  (warning) --quant nf4 requested but this dir is not pre-quantized. "
                      "Either use a different --model_path or switch quants.")
    else:
        print("  LLM: no config.json found")

    for sub in ("image_tokenizer", "decoder", "decoder-turbo", "vae"):
        sp = os.path.join(p, sub)
        if os.path.isdir(sp):
            sz = sum(f.stat().st_size for f in os.scandir(sp) if f.is_file())
            print(f"  {sub}/: present ({sz / 1e9:.2f} GB)")
        else:
            print(f"  {sub}/: MISSING")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llada",
        description="LLaDA-2.0-Uni unified CLI (T2I / MMU / edit)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def _common(sub):
        sub.add_argument("--quant", choices=["nf4", "fp8", "bf16"], default=DEFAULT_QUANT,
                         help="LLM backbone precision. nf4=fastest load+lowest VRAM, "
                              "fp8=highest quality but ~200s inline quant, bf16=full-precision.")
        sub.add_argument("--model_path", default=None,
                         help="Override auto-selected dir (normally picked from --quant). "
                              "Must contain LLM + image_tokenizer/ + decoder/ + decoder-turbo/ + vae/.")
        sub.add_argument("--output", default=None, help="Output path (auto-generated if omitted)")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--device", default="cuda")

    subs = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # t2i
    t2i = subs.add_parser("t2i", help="Text-to-image")
    _common(t2i)
    t2i.add_argument("prompt", help="Text prompt to render")
    t2i.add_argument("--llm_steps", type=int, default=8, help="dLLM denoising steps")
    t2i.add_argument("--cfg_scale", type=float, default=2.0)
    t2i.add_argument("--image_h", type=int, default=1024,
                     help="Target image size in pixels (matches the model's README "
                          "sample). Model halves this internally to get the VQ grid, "
                          "decoder's resolution_multiplier=2 doubles back: "
                          "1024 → 1024 px output, 512 → 512 px output.")
    t2i.add_argument("--image_w", type=int, default=1024,
                     help="See --image_h.")
    t2i.add_argument("--decoder_mode", choices=["turbo", "normal"], default="turbo",
                     help="turbo = 8-step distilled decoder (default, ~6× faster, fixed "
                          "by overriding upstream cfg_scale=0/stoch=1 bug). "
                          "normal = full 50-step ODE, same quality but slower.")
    t2i.add_argument("--decoder_steps", type=int, default=None,
                     help="Override default (8 turbo / 50 normal)")
    t2i.add_argument("--resolution_multiplier", type=int, default=2,
                     help="Upscale factor in the decoder (2 = 1024 from 512 tokens)")
    t2i.add_argument("--save_vq_tokens", default=None, metavar="PATH",
                     help="Dump LLM VQ tokens + metadata to this .pt file. "
                          "Replay with `llada decode PATH` to iterate on the "
                          "decoder without paying the LLM cost again.")
    t2i.set_defaults(func=cmd_t2i)

    # mmu
    mmu = subs.add_parser("mmu", help="Image understanding (VQA / captioning)")
    _common(mmu)
    mmu.add_argument("image", help="Path to input image")
    mmu.add_argument("--question", default="Describe this image in detail.")
    mmu.add_argument("--steps", type=int, default=32, help="dLLM denoising steps")
    mmu.add_argument("--block_length", type=int, default=32)
    mmu.add_argument("--gen_length", type=int, default=256)
    mmu.set_defaults(func=cmd_mmu)

    # edit
    ed = subs.add_parser("edit", help="Image editing")
    _common(ed)
    ed.add_argument("image", help="Path to source image")
    ed.add_argument("instruction", help="Natural-language edit instruction")
    ed.add_argument("--input_size", type=int, default=1024,
                    help="Target size (px, one side) used to center-crop the input "
                         "image before tokenization. Larger = sharper edit output but "
                         "quadratically more LLM compute.")
    ed.add_argument("--steps", type=int, default=8)
    ed.add_argument("--block_length", type=int, default=32)
    ed.add_argument("--cfg_text_scale", type=float, default=4.0)
    ed.add_argument("--cfg_image_scale", type=float, default=0.0)
    ed.add_argument("--decoder_mode", choices=["turbo", "normal"], default="turbo",
                    help="turbo = 8-step distilled decoder (default, ~6× faster). "
                         "normal = 50-step ODE, same quality but slower.")
    ed.add_argument("--decoder_steps", type=int, default=None)
    ed.add_argument("--resolution_multiplier", type=int, default=2)
    ed.add_argument("--save_vq_tokens", default=None, metavar="PATH",
                    help="Dump edited VQ tokens + metadata to this .pt file. "
                         "Replay with `llada decode PATH` to iterate on the decoder.")
    ed.set_defaults(func=cmd_edit)

    # decode
    dec = subs.add_parser("decode",
                          help="Replay decoder on a saved VQ-tokens .pt file")
    _common(dec)
    dec.add_argument("tokens_file", help=".pt file written by --save_vq_tokens")
    dec.add_argument("--decoder_mode", choices=["turbo", "normal"], default="turbo")
    dec.add_argument("--decoder_steps", type=int, default=None)
    dec.add_argument("--resolution_multiplier", type=int, default=2)
    dec.set_defaults(func=cmd_decode)

    # info
    info = subs.add_parser("info", help="Print detected model variants at --model_path")
    info.add_argument("--quant", choices=["nf4", "fp8", "bf16"], default=DEFAULT_QUANT)
    info.add_argument("--model_path", default=None)
    info.set_defaults(func=cmd_info)

    return p


def _resolve_model_path(args: argparse.Namespace) -> None:
    """If --model_path wasn't given, pick the canonical dir for --quant."""
    if getattr(args, "model_path", None):
        return
    args.model_path = MODEL_PATHS[args.quant]


def main() -> None:
    args = build_parser().parse_args()
    _resolve_model_path(args)
    args.func(args)


if __name__ == "__main__":
    main()
