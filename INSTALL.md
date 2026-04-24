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

Two-step install — `flash-attn`'s setup.py imports `torch` during build,
which fails if pip is building it in an isolated env where torch isn't
available yet.

```bash
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.8.3
```

### Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` while building
  flash-attn** — you forgot `--no-build-isolation`, or torch isn't
  installed in the active env yet.
- **`transformers` 5.x auto-resolved** — the pin is on line 11 of
  `requirements.txt`; if yours is older the repo's `trust_remote_code`
  loader will fail with cryptic import errors.
- **torchao "cpp extensions skipped" warning** — harmless; FP8 uses
  `torch._scaled_mm` which is in the torch wheel. Only Triton-based
  kernels need the C++ extensions.

## 3. Model weights

Pick one of:

### NF4 (recommended, ~9.5 GB VRAM, ~75 s load)

Pre-quantized artifact ready to load:

```bash
pip install huggingface_hub[cli]
huggingface-cli download SanDiegoDude/LLaDA2.0-Uni-nf4 \
    --local-dir ./models/LLaDA2.0-Uni-nf4
```

### bf16 (full precision, ~32 GB VRAM — won't fit on a 24 GB card)

```bash
huggingface-cli download inclusionAI/LLaDA2.0-Uni \
    --local-dir ./models/LLaDA2.0-Uni
```

### FP8 (inline, ~17 GB VRAM, ~200 s load)

Uses the bf16 weights above, quantized in-place at load time with
`torchao`. No separate download required.

Tell the CLI/UI where the weights are by editing `scripts/llada.py`
`MODEL_PATHS` once, or passing `--model_path` each call.

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

#### Other useful UI flags

- `--no-eager-load` — don't load the LLM at startup (defer to first
  request). Makes startup instant but the first generation pays the
  full reload cost.
- `--share` — expose a public `gradio.live` URL.
- `--quant {nf4,fp8,bf16}` — swap backbone precision. You need the
  matching weights on disk.

## 5. What's different from upstream

- `fix(decoder)` — turbo decoder is now usable (was producing
  grid-streak artifacts on every hardware we tried). Fixes an
  `cfg_scale=0.0` / `stochast_ratio=1.0` default pair in
  `decoder/decode.py`.
- `scripts/llada.py` — unified CLI with t2i / mmu / edit / decode-replay
  / info subcommands, quant-aware model loading, auto output paths.
- `scripts/ui.py` — Gradio UI with persistent pipeline, component-level
  caching, and a `--low_vram` mode for 24 GB cards.
- `scripts/quantize_nf4.py` — one-shot script to produce the NF4
  artifact from the upstream bf16 weights.
