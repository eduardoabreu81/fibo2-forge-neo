# FIBO 2 support for Forge Neo
# Masked edits. Bria's pipeline (diffusers' BriaFibo2Pipeline, Apache-2.0, image_processor.grey_out) greys out the
# region a mask marks in the one picture to edit before it is encoded, for the VAE and for Qwen3-VL alike: a
# (128, 128, 128) grey through the mask made binary at 128. Here the mask is img2img's inpaint mask, as Forge has
# prepared it for the picture (inverted, cropped for "Only masked", resized), handed over by the extension's script
# (scripts/fibo2.py) and kept until the next generation starts. A picture of other proportions than the result's is
# fitted inside it and the sides around it are masked the same way, for the edit to fill them in, as Bria's
# outpainting template does (ImagePadForOutpaint, then the edit with its mask).
#
# No Forge imports: the CPU tests run it as is.

import numpy as np
import torch

GREY = 128 / 255
THRESHOLD = 128
# how far the sides an edit fills in reach into the picture, so the seam is drawn anew
OVERLAP = 8

_mask = None


def set_mask(mask) -> None:
    """The inpaint mask of the generation starting (a PIL image, white where the picture changes), or None."""
    global _mask
    _mask = mask


def current():
    return _mask


def fill_mask(size, box, overlap: int = OVERLAP):
    """A mask of size (width, height), white outside box (left, top, right, bottom): the sides around a picture
    that an edit fills in, reaching overlap pixels into the picture on each side that has one."""
    from PIL import Image

    width, height = size
    left, top, right, bottom = box
    inner = (left + overlap if left > 0 else 0, top + overlap if top > 0 else 0,
             right - overlap if right < width else width, bottom - overlap if bottom < height else height)
    mask = Image.new("L", size, 255)
    mask.paste(0, inner)
    return mask


def grey_out(image: torch.Tensor, mask) -> torch.Tensor:
    """image (1, H, W, 3) in [0, 1] with the region the mask marks, white on black, grey; the mask is brought to the
    picture's size first."""
    from PIL import Image

    height, width = image.shape[1], image.shape[2]
    mask = mask.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
    region = torch.from_numpy(np.array(mask) >= THRESHOLD)[None, :, :, None]
    return torch.where(region, torch.full_like(image, GREY), image)
