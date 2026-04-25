# Install & Run

## 1. Environment

Python 3.12 is required (3.14 is too new for PyTorch wheels, 3.11 and
older miss some `transformers==4.57.3` features).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Dependencies

### 2a. Install PyTorch FIRST, matching your CUDA driver

Do **not** run `pip install -r requirements.txt` yet. The default PyPI
index will pull torch 2.11 which ships a CUDA 13 runtime — that works
only with driver 580+ and has no prebuilt `flash-attn` wheel, so the
build falls over with:

```
RuntimeError: The detected CUDA version (12.x) mismatches the version
that was used to compile PyTorch (13.0).
```

Pick one based on `nvidia-smi`:

| GPU / driver | CUDA reported | Install command |
|---|---|---|
| DGX Spark / Blackwell | 13.0 | `pip install torch --index-url https://download.pytorch.org/whl/cu130` |
| RTX 4090 / 3090 / 4080 | 12.8 | `pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128` |
| Older consumer cards | 12.4 | `pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124` |

Verify before moving on:

```bash
python -c "import torch; print('cuda ok:', torch.cuda.is_available(), 'torch cuda:', torch.version.cuda)"
```

### 2b. Install the rest

```bash
pip install -r requirements.txt
```

`requirements.txt` caps torch below 2.11 so this step can't silently
upgrade past a working cu12 build.

### 2c. Install flash-attn

Two-step is required because `flash-attn`'s `setup.py` imports `torch`
at build time, which fails in an isolated pip-build env where torch
isn't installed yet.

```bash
pip install --no-build-isolation flash-attn==2.8.3
```

This downloads a prebuilt wheel matching `(torch, cuda, python)` from
`https://github.com/Dao-AILab/flash-attention/releases`. If the exact
combo isn't there, pip falls back to a source build (needs `nvcc` on
`$PATH` and matching `CUDA_HOME`).

### Troubleshooting

- **`RuntimeError: The detected CUDA version (12.x) mismatches the
  version that was used to compile PyTorch (13.0).`** — you installed
  torch from the default PyPI index (which picked cu13). Uninstall
  torch and torchvision, then reinstall from the correct cu12 index
  per 2a.
- **`ModuleNotFoundError: No module named 'torch'` while building
  flash-attn** — you forgot `--no-build-isolation`, or torch isn't
  installed in the active env yet.
- **`urllib.error.HTTPError: HTTP Error 404: Not Found` during
  flash-attn build, URL contains `cu12torch2.11`** — a prebuilt wheel
  for your torch version doesn't exist upstream. Pin to the torch
  version listed in 2a (2.7.1 for cu128, 2.5.1 for cu124).
- **`transformers` 5.x auto-resolved** — the pin is in
  `requirements.txt`. If yours is newer the `trust_remote_code` loader
  fails with cryptic import errors.
- **torchao "cpp extensions skipped" warning** — harmless; FP8 uses
  `torch._scaled_mm` which is in the torch wheel. Only Triton-based
  kernels need the C++ extensions.
- **Browser tab freezes / crashes when switching Gradio tabs** — you
  have Gradio 6.11+. A Svelte reactivity loop (gradio#13198) locks up
  the frontend. We pin to `gradio==6.10.0` in requirements.txt; if
  you upgrade past that you'll reintroduce the bug.

## 3. Model weights

**New:** weights auto-download on first use. Just run any command — the
CLI will grab the matching repo from Hugging Face into `./models/` and
continue. You can override the cache root with:

```bash
export LLADA_MODELS_DIR=/path/to/big/disk
```

Or pre-fetch explicitly:

```bash
python scripts/llada.py download --quant nf4     # 9 GB,  recommended
python scripts/llada.py download --quant bf16    # 40 GB, needed for fp8 too
```

| Quant | Source repo | Size | VRAM (load) |
|---|---|---|---|
| `nf4` | `SanDiegoDude/LLaDA2.0-Uni-bnb-nf4` | ~9 GB | ~9.5 GB |
| `bf16` | `inclusionAI/LLaDA2.0-Uni` | ~40 GB | ~32 GB |
| `fp8` | `inclusionAI/LLaDA2.0-Uni` (inline-quantized at load) | ~40 GB | ~17 GB |

## 4. Run

### Unified CLI — one-off generations, good for scripting

```bash
# Text-to-image
python scripts/llada.py t2i \
    "a red panda wearing sunglasses sitting on a bamboo chair" \
    --quant nf4 --output outputs/panda.png

# Image understanding
python scripts/llada.py mmu ./assets/my_image.png \
    --question "What's happening in this photo?"

# Image editing
python scripts/llada.py edit ./my_image.png \
    "replace the background with a snowy mountain" \
    --output outputs/edited.png

# Full help
python scripts/llada.py --help
python scripts/llada.py t2i --help
```

### Gradio UI — persistent model, much faster iteration

```bash
python scripts/ui.py --quant nf4 --host 0.0.0.0 --port 7860
# Then open http://localhost:7860
```

#### On a 24 GB card (RTX 4090 / 3090)

Add `--low_vram` — the LLM and decoder are then swapped on and off the
GPU so they never co-reside. Peak VRAM stays around 14 GB at the cost
of ~75 s of LLM reload between generations. The "Replay decoder" tab
in the UI does not pay this cost because only the decoder is resident
during that loop.

```bash
python scripts/ui.py --quant nf4 --low_vram --host 0.0.0.0 --port 7860
```

#### Load-on-demand (`--lod`)

For a long-running server that shouldn't hog the GPU when idle:

```bash
python scripts/ui.py --quant nf4 --lod --host 0.0.0.0 --port 7860
```

After each request completes, every component (LLM, decoder, VAE,
image tokenizer, SigVQ) is unloaded from GPU memory. Between
requests the process sits at ~0 GB VRAM. Every request pays the full
~90 s cold-start penalty. `--lod` is stronger than `--low_vram` and
implies it. Use this when you want to park the server on a shared box
and still keep the GPU free for other workloads between jobs.

#### HTTP API (`--api`, pairs with `--lod`)

```bash
python scripts/ui.py --quant nf4 --lod --api --host 0.0.0.0 --port 7860
```

Now, on the same port:

- `http://host:7860/ui`   — the Gradio UI
- `http://host:7860/api/` — REST endpoints (FastAPI)
- `http://host:7860/docs` — auto-generated OpenAPI explorer

Endpoints (all POST JSON, return JSON; images are base64-encoded PNG
inline, nothing is written to disk on the server):

| Path | What it does |
|---|---|
| `POST /api/t2i` | `{prompt, image_h, image_w, llm_steps, cfg_scale, decoder_mode, decoder_steps, seed}` → `{image_base64, meta, timing}` |
| `POST /api/edit` | `{image_base64, instruction, input_size, llm_steps, block_length, cfg_text_scale, cfg_image_scale, decoder_mode, decoder_steps, seed}` → `{image_base64, meta, timing}` |
| `POST /api/mmu` | `{image_base64, question, llm_steps, block_length, gen_length, seed}` → `{answer, timing}` |
| `GET /api/status` | pipeline state, loaded components, CUDA memory |

`decoder_mode` is `"turbo"` (default, 8-step distilled) or `"normal"`
(50-step, higher fidelity, slower). `decoder_steps` overrides the mode
default if provided. Everything else has a sensible default; the
minimal t2i call is just `{"prompt": "..."}`.

Quick test:

```bash
curl -X POST http://localhost:7860/api/t2i \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a neon-lit cyberpunk fox, cinematic"}' \
  | python -c "import sys,json,base64; d=json.loads(sys.stdin.read()); open('out.png','wb').write(base64.b64decode(d['image_base64'])); print('meta:', d['meta']); print('timing:', d['timing'])"
```

#### Other useful UI flags

- `--no-eager-load` — don't load the LLM at startup (defer to first
  request). Makes startup instant but the first generation pays the
  full reload cost. Implied by `--lod`.
- `--share` — expose a public `gradio.live` URL (UI-only, no effect
  when `--api` is set — use a reverse proxy for public API access).
- `--quant {nf4,fp8,bf16}` — global backbone precision applied to both
  LLM and decoder. Weights for the requested quant auto-download on
  first use.
- `--llm-quant {nf4,fp8,bf16}` / `--decoder-quant {nf4,fp8,bf16}` —
  override per-component. Whatever you don't specify falls back to
  `--quant`. Mix-and-match example:

  ```bash
  python scripts/ui.py --llm-quant fp8 --decoder-quant nf4 --low_vram
  ```

  Note: the released NF4 and bf16 model dirs ship _identical_ bf16
  decoder weights, so today this is mostly forward-compatible plumbing
  and only affects which dir the decoder safetensors load from. The
  knob is wired so a future quantized decoder slots in cleanly.

## 5. What's different from upstream

- `fix(decoder)` — turbo decoder is now usable (was producing
  grid-streak artifacts on every hardware we tried). Fixes a
  `cfg_scale=0.0` / `stochast_ratio=1.0` default pair in
  `decoder/decode.py`, and a `time_shifting_factor` bug in the
  stochastic branch of `decoder/transport/transport.py`.
- `scripts/llada.py` — unified CLI with `t2i` / `mmu` / `edit` /
  `decode` / `download` / `info` subcommands, quant-aware model
  loading, auto-download from Hugging Face, auto output paths.
- `scripts/ui.py` — Gradio UI with persistent pipeline, component-level
  caching, and a `--low_vram` mode for 24 GB cards.
- `scripts/quantize_nf4.py` — one-shot script to produce the NF4
  artifact from the upstream bf16 weights.
