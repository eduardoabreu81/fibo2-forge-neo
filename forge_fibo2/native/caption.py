# FIBO 2 support for Forge Neo
# The prompt as FIBO 2.0 was trained on it, and the pictures Qwen3-VL sees, ported from ComfyUI's
# comfy/text_encoders/fibo2.py and comfy_extras/nodes_fibo2.py (COMFYUI_COMMIT in native/__init__.py), which follow
# Bria's reference pipeline.
#
# A prompt is a structured JSON caption (Bria's VGL) or plain text. The caption is cleaned as in training (empty
# fields and scores dropped) and re-serialized compactly. Each text_render entry gets a marker where Qwen3-VL sees a
# picture of its text, drawn black on white (text_images.py). An edit wraps the caption, or the plain text as the
# instruction, in
# {"image": <marker>, ..., "structured_caption": ..., "edit_instruction": ...}. An empty prompt is the null caption
# training used for classifier-free guidance.
#
# No Forge imports: the CPU tests run it as is.

import json
import math

import numpy as np
import torch

from . import text_images

VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
# the null caption, a whole chat turn, taken as it is (no cleaning, no template)
NULL_CAPTION = TEMPLATE.format('{"short_description":"","objects":[],"background_setting":"","lighting":{},'
                               '"aesthetics":{},"photographic_characteristics":{},"style_medium":"","context":"",'
                               '"artistic_style":""}')
SCORES = ("aesthetic_score", "preference_score")
# Qwen3-VL sees each image to edit at most this many pixels a side, as in training
THUMBNAIL = 768


def dumps(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def empty(value):
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def clean(value):
    if isinstance(value, dict):
        value = {k: clean(v) for k, v in value.items()}
        return {k: v for k, v in value.items() if not empty(v)}
    if isinstance(value, list):
        return [v for v in map(clean, value) if not empty(v)]
    return value


def parse_prompt(prompt):
    """(caption, edit instruction): the caption as training saw it; a prompt that isn't a JSON object comes back as
    it is, with no instruction."""
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction

    keys = set(caption)
    caption = clean({key: {k: v for k, v in value.items() if k not in SCORES} if isinstance(value, dict) else value
                     for key, value in caption.items() if key not in SCORES})
    # exactly a short_description and a text_render list keeps both, the list even when empty: the image has no text
    if keys == {"short_description", "text_render"}:
        caption = {"short_description": caption.get("short_description") or "",
                   "text_render": caption.get("text_render") or []}
    return caption or None, instruction


def build(prompt: str, edit_images: int = 0, text_images: bool = True) -> tuple[str, list[str]]:
    """The chat turn for a prompt, and the texts to draw as pictures, in the order of their markers after the
    edit images'. text_images=False leaves each text entry's marker empty: the caption a generation starts from."""
    if prompt.strip() == "":
        return NULL_CAPTION, []
    caption, instruction = parse_prompt(prompt)
    if edit_images > 0 and instruction is None:
        caption, instruction = None, prompt
    entries = []
    if isinstance(caption, dict):
        entries = [e for e in caption.get("text_render") or []
                   if isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"]]
    for entry in entries:
        entry["image_tokens"] = VISION_MARKER if text_images else ""

    if edit_images > 0:
        edit = ({"image": VISION_MARKER} if edit_images == 1
                else {f"image_{k}": VISION_MARKER for k in range(1, edit_images + 1)})
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        text = dumps(edit)
    elif not isinstance(caption, str):
        text = dumps(caption or {})
    else:
        text = caption
    texts = [entry["text"] for entry in entries] if text_images else []
    return (text if text.startswith("<|im_start|>") else TEMPLATE.format(text)), texts


def has_text_render(prompt: str) -> bool:
    return bool(build(prompt, text_images=True)[1])


# pictures: (1, H, W, 3) float tensors in [0, 1], as ComfyUI's images


def text_picture(text: str) -> torch.Tensor:
    """The picture of a text_render text, drawn as training drew it (text_images.py)."""
    image = text_images.draw_text_image(text)
    return torch.from_numpy(np.array(image)).float().div(255.0).unsqueeze(0)


def resize_8bit(image, height, width):
    # antialiased bicubic rounded to 8 bit, as PIL and Qwen3-VL's processor resize
    image = torch.nn.functional.interpolate(image.movedim(-1, 1), size=(height, width), mode="bicubic", antialias=True)
    return image.mul(255.0).round().clamp(0, 255).div(255.0).movedim(1, -1)


def thumbnail(image, size=THUMBNAIL):
    scale = size / max(image.shape[1], image.shape[2])
    if scale >= 1:
        return image
    return resize_8bit(image, max(1, round(image.shape[1] * scale)), max(1, round(image.shape[2] * scale)))


def to_qwen_grid(image):
    # the processor's 32 pixel grid, so Forge's own bilinear resize to it is a no-op
    h, w = image.shape[1], image.shape[2]
    height, width = round(h / 32) * 32, round(w / 32) * 32
    if height * width > 1605632:
        beta = math.sqrt(h * w / 1605632)
        height, width = max(32, math.floor(h / beta / 32) * 32), max(32, math.floor(w / beta / 32) * 32)
    elif height * width < 3136:
        beta = math.sqrt(3136 / (h * w))
        height, width = math.ceil(h * beta / 32) * 32, math.ceil(w * beta / 32) * 32
    return resize_8bit(image, height, width)


def trained_size(aspect_ratio, step=16):
    """Bria's (width, height) for an aspect ratio: of the multiples of 16 within 10% of 1 MP, the least cropping,
    then the closest to 1 MP. Another step gives the same choice among its multiples (Forge's Resolution Step)."""
    area = 1024 * 1024
    sizes = [(w, h) for w in range(64, 2 * area // 64, step)
             for h in range(max(64, step * math.ceil(0.9 * area / w / step)), step * math.floor(1.1 * area / w / step) + 1,
                            step)]
    return min(sizes, key=lambda s: (crop(s[0] / s[1], aspect_ratio), abs(s[0] * s[1] - area), -s[0]))


def crop(aspect_ratio, other):
    # the share of a picture of one aspect ratio cropped away to fit the other
    return round(1 - min(aspect_ratio / other, other / aspect_ratio), 9)


def same_proportions(picture, size):
    # within 5%: the share cropped away to fit one into the other
    return crop(size[0] / size[1], picture[0] / picture[1]) <= 0.05


def size_advice(picture, size, step=16):
    """A better (width, height) for editing a picture of size (width, height) into a result of size, or None when
    size is fine: about 1 MP (within 10%) and the picture's proportions (within 5%), as Edit Turbo was trained.
    Elsewhere edits tend to zoom in on what changes, or lose detail."""
    if 0.9 <= size[0] * size[1] / (1024 * 1024) <= 1.1 and same_proportions(picture, size):
        return None
    return trained_size(picture[0] / picture[1], step)


def fill_box(picture, size):
    """(left, top, right, bottom) of a picture of size (width, height) scaled to fit inside a result of size and
    centered, where Forge's "Resize and fill" puts it (modules/images.py, resize_image mode 2)."""
    width, height = size
    ratio, picture_ratio = width / height, picture[0] / picture[1]
    inner_width = width if ratio < picture_ratio else picture[0] * height // picture[1]
    inner_height = height if ratio >= picture_ratio else picture[1] * width // picture[0]
    left, top = width // 2 - inner_width // 2, height // 2 - inner_height // 2
    return left, top, left + inner_width, top + inner_height


def fit(image, width, height):
    """Scaled and center cropped to width x height with Lanczos, as ComfyUI's common_upscale(..., "lanczos",
    "center"); unchanged when it already is that size."""
    if (image.shape[2], image.shape[1]) == (width, height):
        return image
    from PIL import Image

    old_height, old_width = image.shape[1], image.shape[2]
    old_aspect, new_aspect = old_width / old_height, width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
    cropped = image[0, y:old_height - y, x:old_width - x]
    pil = Image.fromarray(np.clip(255.0 * cropped.cpu().numpy(), 0, 255).astype(np.uint8))
    pil = pil.resize((width, height), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)


def pad(image, width, height):
    """Scaled to fit inside width x height with Lanczos and centered on the (128, 128, 128) grey of a masked edit,
    for the edit to fill the sides in; unchanged when it already is that size."""
    if (image.shape[2], image.shape[1]) == (width, height):
        return image
    from PIL import Image

    left, top, right, bottom = fill_box((image.shape[2], image.shape[1]), (width, height))
    pil = Image.fromarray(np.clip(255.0 * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
    pil = pil.resize((right - left, bottom - top), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    canvas.paste(pil, (left, top))
    return torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)


def edit_images(images, width=None, height=None, fill=False):
    """The images to edit at the sizes upstream gives them: a single one scaled and center cropped to the result's
    size, width x height (multiples of 16; its own size when not given), since the edit lines up with it pixel for
    pixel, or with fill fitted inside it on grey; several, each to its own trained size."""
    images = [image[..., :3] for image in images]
    if len(images) == 1:
        h, w = images[0].shape[1], images[0].shape[2]
        width, height = width or w, height or h
        return [(pad if fill else fit)(images[0], max(16, width // 16 * 16), max(16, height // 16 * 16))]
    return [fit(image, *trained_size(image.shape[2] / image.shape[1])) for image in images]
