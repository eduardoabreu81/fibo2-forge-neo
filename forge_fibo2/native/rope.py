# FIBO 2 support for Forge Neo
# The RoPE scale of "[FIBO 2] RoPE scale": ComfyUI's ScaleROPE node (comfy_extras/nodes_rope.py) set from the
# settings. The image's position ids step by the scale instead of 1 (transformer.image_ids), so a picture larger than
# the model was trained on spans the position range it knows: no stretched bodies or repeated subjects, at some cost
# in fine detail. Linear position interpolation, the same scale on both axes and on the reference pictures, as
# ComfyUI applies it.
#
# Auto sets the scale from the picture's area: 1.0 up to REFERENCE_TOKENS, then the square root of the area ratio,
# so the positions span the reference area. FIBO 2 takes a modest stretch past its 1 MP training size as it is;
# 1.5 MP gives 0.82 at 1536x1536, where users report 0.8 by hand.
#
# No Forge imports: the CPU tests run it as is.

import math

PATCH = 16
MIN_SCALE, MAX_SCALE = 0.5, 1.0
# 1.5 MP in 16x16 pixel tokens: the largest area Auto leaves unscaled
REFERENCE_TOKENS = 6144


def auto_scale(width: int, height: int) -> float:
    """The Auto scale for a width x height picture, rounded to the slider's 0.01 step."""
    tokens = (int(height) // PATCH) * (int(width) // PATCH)
    if tokens <= REFERENCE_TOKENS:
        return MAX_SCALE
    return clamp(round(math.sqrt(REFERENCE_TOKENS / tokens), 2))


def clamp(scale: float) -> float:
    return min(MAX_SCALE, max(MIN_SCALE, float(scale)))


def with_scale(transformer_options: dict | None, scale: float) -> dict | None:
    """transformer_options with the scale as ComfyUI's rope_options; unchanged at 1.0 (today's positions, byte for
    byte) or when another patch has set rope_options already."""
    if scale == 1.0 or (transformer_options or {}).get("rope_options"):
        return transformer_options
    options = dict(transformer_options or {})
    options["rope_options"] = {"scale_x": scale, "scale_y": scale}
    return options
