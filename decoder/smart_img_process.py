"""Smart image resizing with aspect-ratio preservation and factor alignment."""

import math
from typing import List, Tuple
from PIL import Image


def smart_resize(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = 32,
) -> Tuple[int, int]:
    """
    Qwen2.5-VL style smart resize.
    Scales the image to fit within [min_pixels, max_pixels] while preserving
    the aspect ratio, and returns target dimensions aligned to ``factor``.
    """
    h_bar = max(round(height / factor) * factor, factor)
    w_bar = max(round(width / factor) * factor, factor)

    if h_bar * w_bar > max_pixels:
        scale = math.sqrt(max_pixels / (height * width))
        h_bar = max(math.floor(height * scale / factor) * factor, factor)
        w_bar = max(math.floor(width * scale / factor) * factor, factor)
    elif h_bar * w_bar < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * scale / factor) * factor
        w_bar = math.ceil(width * scale / factor) * factor

    return h_bar, w_bar


def resize_and_center_crop(
    img: Image.Image,
    target_h: int,
    target_w: int,
    factor: int = 32,
) -> Image.Image:
    """
    Resize the image (preserving aspect ratio) so that it covers the target
    dimensions, then center-crop to a factor-aligned size.
    """
    width, height = img.size

    # Scale so the image just covers the target area
    scale = max(target_h / height, target_w / width)
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))

    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # Center-crop to factor-aligned dimensions
    crop_h = (new_h // factor) * factor
    crop_w = (new_w // factor) * factor

    # Ensure at least target size
    crop_h = max(crop_h, target_h)
    crop_w = max(crop_w, target_w)

    top = (new_h - crop_h) // 2
    left = (new_w - crop_w) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h))

    return img


def smart_resize_images(
    image_paths: List[str],
    patch_size: int = 16,
    merge_size: int = 2,
    single_min_pixels: int = 128 * 128,
    single_max_pixels: int = 800 * 800,
    multi_min_pixels: int = 128 * 128,
    multi_max_pixels: int = 448 * 448,
) -> List[Image.Image]:
    """
    Smart-resize a list of images for model input.

    Uses larger resolution limits for single-image inputs and smaller limits
    for multi-image inputs to control total token count.
    """
    num_images = len(image_paths)
    if num_images == 0:
        return []

    factor = patch_size * merge_size  # 32

    if num_images == 1:
        min_pixels = single_min_pixels
        max_pixels = single_max_pixels
    else:
        min_pixels = multi_min_pixels
        max_pixels = multi_max_pixels

    images = []

    for path in image_paths:
        if path is None:
            images.append(path)
            continue
        img = Image.open(path).convert("RGB")
        width, height = img.size

        target_h, target_w = smart_resize(height, width, min_pixels, max_pixels, factor)

        img = resize_and_center_crop(img, target_h, target_w, factor)
        images.append(img)

    return images
