"""
Pre-quantize LLaDA2.0-Uni backbone to NF4 and save to disk.

Output layout (dst_dir):
  config.json                         # with embedded quantization_config
  model-*.safetensors                 # NF4-packed backbone (~9 GB total)
  model.safetensors.index.json
  modeling_llada2uni_moe.py           # custom code (copied)
  configuration_llada2uni_moe.py
  tokenizer.json / special_tokens_map / tokenizer_config
  image_tokenizer/                    # copied as-is (stays bf16)
  decoder/                            # copied as-is (stays bf16)
  decoder-turbo/                      # copied as-is (stays bf16)
  vae/                                # copied as-is
  assets/                             # optional
  README.md

Run:
  python scripts/quantize_nf4.py --src /path/to/LLaDA2.0-Uni --dst /path/to/LLaDA2.0-Uni-nf4
"""

import argparse
import os
import shutil
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True)


AUX_DIRS = ["image_tokenizer", "decoder", "decoder-turbo", "vae", "assets"]
AUX_FILES_COPY_IF_EXIST = [
    "modeling_llada2uni_moe.py",
    "configuration_llada2uni_moe.py",
    "README.md",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True, help="Source model dir (bf16)")
    p.add_argument("--dst", type=str, required=True, help="Destination dir for NF4 artifact")
    p.add_argument("--quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--double_quant", action="store_true", default=True)
    p.add_argument("--compute_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16"])
    p.add_argument("--skip_modules", nargs="+",
                   default=["lm_head", "gate"],
                   help="Modules to keep in full precision")
    p.add_argument("--copy_aux", action="store_true", default=True,
                   help="Copy image_tokenizer/decoder/decoder-turbo/vae alongside")
    p.add_argument("--no_copy_aux", dest="copy_aux", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    compute_dtype = torch.bfloat16 if args.compute_dtype == "bfloat16" else torch.float16

    os.makedirs(args.dst, exist_ok=True)

    # --- Load model inline-quantized ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=args.double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=args.skip_modules,
    )

    print(f"[1/4] Loading {args.src} with NF4 quant "
          f"(quant_type={args.quant_type}, dq={args.double_quant}, "
          f"compute={args.compute_dtype}, skip={args.skip_modules})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.src,
        quantization_config=bnb,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map={"": "cuda"},
    ).eval()
    t_load = time.time() - t0
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"      loaded in {t_load:.1f}s, CUDA mem: {mem_gb:.2f} GB")

    import bitsandbytes as bnb_mod
    n_4 = sum(1 for m in model.modules() if isinstance(m, bnb_mod.nn.Linear4bit))
    print(f"      Linear4bit modules: {n_4}")

    # --- Save quantized backbone ---
    print(f"[2/4] Saving quantized weights to {args.dst} ...")
    t0 = time.time()
    # safe_serialization=True uses safetensors; max_shard_size keeps shards manageable
    model.save_pretrained(args.dst, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.dst)
    print(f"      saved in {time.time() - t0:.1f}s")

    # Release GPU
    del model
    torch.cuda.empty_cache()

    # --- Copy custom code files + README ---
    print("[3/4] Copying custom code files...")
    for fname in AUX_FILES_COPY_IF_EXIST:
        src = os.path.join(args.src, fname)
        dst = os.path.join(args.dst, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"      + {fname}")

    # --- Copy auxiliary directories (image_tokenizer, decoder, etc.) ---
    if args.copy_aux:
        print("[4/4] Copying auxiliary directories (this may take a while for decoder/*)...")
        for d in AUX_DIRS:
            src = os.path.join(args.src, d)
            dst = os.path.join(args.dst, d)
            if not os.path.isdir(src):
                print(f"      - {d}: not present in source, skipping")
                continue
            if os.path.exists(dst):
                print(f"      - {d}: already exists at dst, skipping")
                continue
            t0 = time.time()
            shutil.copytree(src, dst, copy_function=shutil.copy2)
            sz = sum(f.stat().st_size for f in os.scandir(dst) if f.is_file()) / 1e9
            print(f"      + {d} ({sz:.2f} GB ish, {time.time() - t0:.1f}s)")
    else:
        print("[4/4] Skipping auxiliary dirs (--no_copy_aux)")

    print("\nDone. Quantized artifact at:", args.dst)
    total_bytes = 0
    for root, _, files in os.walk(args.dst):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"Total size on disk: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
