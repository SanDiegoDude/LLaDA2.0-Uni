"""
LLaDA-2.0-Uni — Image Editing

Usage:
    python image_edit.py --model_path /path/to/LLaDA-2.0-Uni --image input.jpg --instruction "Change the background to a beach."
    python image_edit.py --model_path /path/to/LLaDA-2.0-Uni --image_token input.pt --instruction "Make it a watercolor painting."
"""

import os, sys, gc, argparse, torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decoder import decode_vq_tokens


def parse_args():
    p = argparse.ArgumentParser(description="LLaDA-2.0-Uni Image Editing")
    p.add_argument("--model_path", type=str, required=True,
                   help="Root model dir containing LLM weights, image_tokenizer/, decoder/, vae/")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--image_token", type=str, default=None)
    p.add_argument("--instruction", type=str, required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--cfg_text_scale", type=float, default=4.0)
    p.add_argument("--cfg_image_scale", type=float, default=0.0)
    p.add_argument("--decoder_steps", type=int, default=50)
    p.add_argument("--resolution_multiplier", type=int, default=2)
    p.add_argument("--output", type=str, default="edited.png")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _get_image_token_offset(model_path):
    """Read image_token_offset from model config."""
    import json
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f).get("image_token_offset", 157184)


def encode_image_from_pt(pt_path, offset):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    token_ids = (data["semantic_token_ids"] + offset).tolist()
    w, h = data["metadata"]["processed_size"]
    return token_ids, h // 16, w // 16


def encode_image_from_pil(image_path, model_path, device, offset):
    from encoder.image_tokenizer import ImageTokenizer
    from decoder.utils import generate_crop_size_list, var_center_crop

    image_tokenizer = ImageTokenizer(
        model_path=model_path, device=device, dtype=torch.bfloat16,
    )
    crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
    pil_image = var_center_crop(Image.open(image_path).convert("RGB"), crop_size_list=crop_size_list)
    info = image_tokenizer.encode_with_info(pil_image)
    _, h, w = info["grid_thw"]
    token_ids = [x + offset for x in info["token_ids"]]
    del image_tokenizer; torch.cuda.empty_cache()
    return token_ids, h, w


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode source image
    offset = _get_image_token_offset(args.model_path)
    if args.image_token:
        print(f"Loading pre-tokenized image: {args.image_token}")
        image_tokens, image_h, image_w = encode_image_from_pt(args.image_token, offset)
    elif args.image:
        print(f"Encoding image: {args.image}")
        image_tokens, image_h, image_w = encode_image_from_pil(args.image, args.model_path, device, offset)
    else:
        raise ValueError("Provide --image or --image_token")

    print(f"Image grid: {image_h}x{image_w}, instruction: {args.instruction}")

    # Phase 1: generate edited VQ tokens
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map={"": device}, trust_remote_code=True
    ).to(torch.bfloat16).eval()
    model.tokenizer = tokenizer

    result = model.edit_image(
        image_tokens, image_h, image_w, args.instruction,
        steps=args.steps, block_length=args.block_length,
        cfg_text_scale=args.cfg_text_scale, cfg_image_scale=args.cfg_image_scale,
    )

    del model; gc.collect(); torch.cuda.empty_cache()
    print("Model unloaded.\n")

    # Phase 2: decode to image
    print("Decoding edited image...")
    img = decode_vq_tokens(result["token_ids"], result["h"], result["w"],
                           args.model_path, device,
                           resolution_multiplier=args.resolution_multiplier, num_steps=args.decoder_steps)
    img.save(args.output)
    print(f"\n✅ Saved: {args.output}")


if __name__ == "__main__":
    main()
