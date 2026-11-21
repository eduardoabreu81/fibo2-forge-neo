# FIBO 2 support for Forge Neo
# The RoPE scale of "[FIBO 2] RoPE scale": ComfyUI's ScaleROPE node (comfy_extras/nodes_rope.py) set from the
# settings. The image's position ids step by the scale instead of 1 (transformer.image_ids), so a picture larger than
# the model was trained on spans fewer positions. Linear position interpolation, the same scale on both axes and on
# the reference pictures, as ComfyUI applies it.
#
# In Forge the size-dependent shift (shift.py) already keeps large pictures from stretching, which ComfyUI's fixed
# shift does not; here the scale sets the framing. At 1 a larger picture is composed as a wider scene and the
# subjects get smaller; below 1 the framing comes back towards that of 1 MP, the training size. Below about 0.65 the
# positions are too close together and fine structure smears and merges (0.60 at 2.2 MP and 0.63 at 4 MP in tests),
# whatever the picture's size, so the slider stops at MIN_SCALE.
#
# Auto sets the scale from the picture's area: the square root of 1 MP over the area, so the positions span about
# 1 MP, but never below AUTO_FLOOR. 1.0 up to 1 MP, 0.82 at 1.5 MP, AUTO_FLOOR from about 1.56 MP up; 0.75 to 0.80
# looked right at 2.2 and 4 MP.
#
# No Forge imports: the CPU tests run it as is.

import math

PATCH = 16
MIN_SCALE, MAX_SCALE = 0.7, 1.0
AUTO_FLOOR = 0.8
# 1 MP in 16x16 pixel tokens: the area whose framing Auto keeps
REFERENCE_TOKENS = 4096


def auto_scale(width: int, height: int) -> float:
    """The Auto scale for a width x height picture, rounded to the slider's 0.01 step."""
    tokens = (int(height) // PATCH) * (int(width) // PATCH)
    if tokens <= REFERENCE_TOKENS:
        return MAX_SCALE
    return max(AUTO_FLOOR, round(math.sqrt(REFERENCE_TOKENS / tokens), 2))


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
