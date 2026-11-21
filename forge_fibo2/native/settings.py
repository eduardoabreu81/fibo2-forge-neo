# FIBO 2 support for Forge Neo
# Five options on a "FIBO 2" page of their own under Settings -> Extensions, as the MiniMax H3 extension keeps its own.
# Their keys are the ones they had in the "Stable Diffusion" section, so saved values carry over.
# "[FIBO 2] Enable Reference", as Forge Neo's own "[Klein] / [Qwen 2.1] / [Krea2] Enable Reference": on, the img2img
# image (and the ones ImageStitch adds) are images to edit, and the prompt is the edit instruction; off, img2img is
# plain image-to-image.
# "[FIBO 2] Reference Sets the Size": on, a txt2img edit of a single ImageStitch picture comes out about 1 MP in the
# picture's proportions, as Wan2GP's default for Edit Turbo, or, when the sliders ask for other proportions, at
# their size with the sides around the picture filled in; off, at the size of txt2img's sliders, the picture cropped.
# "[FIBO 2] Fused kernels": on, the DiT runs on comfy-kitchen's fused kernels where it can (kernels.py); off, the
# eager path, the same code as before the fused one existed, for A/B pictures and timings and for bug reports.
# "[FIBO 2] RoPE scale" and "[FIBO 2] RoPE scale: Auto": ComfyUI's ScaleROPE (rope.py), for pictures larger than
# the model was trained on; 1.0 leaves the positions as they were. Auto sets the scale from each pass's size and
# ignores the slider.

from . import rope

OPTION = "fibo2_do_reference"
DEFAULT = True
LABEL = "[FIBO 2] Enable Reference"
SIZE_OPTION = "fibo2_reference_size"
SIZE_DEFAULT = True
SIZE_LABEL = "[FIBO 2] Reference Sets the Size"
FUSED_OPTION = "fibo2_fused_kernels"
FUSED_DEFAULT = True
FUSED_LABEL = "[FIBO 2] Fused kernels"
ROPE_OPTION = "fibo2_rope_scale"
ROPE_DEFAULT = 1.0
ROPE_LABEL = "[FIBO 2] RoPE scale"
ROPE_AUTO_OPTION = "fibo2_rope_scale_auto"
ROPE_AUTO_DEFAULT = True
ROPE_AUTO_LABEL = "[FIBO 2] RoPE scale: Auto"
# a page under Settings -> Extensions: a section with no category lands there
SECTION = ("fibo2", "FIBO 2")
# where the user finds them, for console messages
PAGE = "Settings -> Extensions -> FIBO 2"


def _get(option: str, default):
    try:
        from modules.shared import opts

        return getattr(opts, option, default)
    except Exception:  # outside Forge Neo (CPU tests), or before the options exist
        return default


def references_enabled() -> bool:
    return bool(_get(OPTION, DEFAULT))


def reference_sets_size() -> bool:
    return bool(_get(SIZE_OPTION, SIZE_DEFAULT))


def fused_enabled() -> bool:
    return bool(_get(FUSED_OPTION, FUSED_DEFAULT))


def rope_scale(width: int, height: int) -> float:
    """The RoPE scale for a sampling pass at width x height: Auto's from the size, else the slider's."""
    if bool(_get(ROPE_AUTO_OPTION, ROPE_AUTO_DEFAULT)):
        return rope.auto_scale(width, height)
    try:
        return rope.clamp(_get(ROPE_OPTION, ROPE_DEFAULT))
    except (TypeError, ValueError):  # a hand-edited config.json
        return ROPE_DEFAULT


def register() -> None:
    import gradio as gr
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        info = OptionInfo(DEFAULT, LABEL, section=SECTION)
        info = info.info("enable Edit ; disable img2img").info("pin to <b>Quicksettings</b> is recommended if changed often")
        shared.opts.add_option(OPTION, info)
    if SIZE_OPTION not in shared.opts.data_labels:
        info = OptionInfo(SIZE_DEFAULT, SIZE_LABEL, section=SECTION)
        info = info.info("txt2img with one ImageStitch picture: enable about 1 MP in its proportions, or the sliders' "
                         "size with the sides filled in when theirs differ ; disable the sliders' size, cropped")
        shared.opts.add_option(SIZE_OPTION, info)
    if FUSED_OPTION not in shared.opts.data_labels:
        info = OptionInfo(FUSED_DEFAULT, FUSED_LABEL, section=SECTION)
        info = info.info("comfy-kitchen fused norms, RoPE and SwiGLU in the DiT (bf16 and INT8 ConvRot files on CUDA) ; "
                         "off runs the plain path, for comparisons")
        shared.opts.add_option(FUSED_OPTION, info)
    if ROPE_OPTION not in shared.opts.data_labels:
        info = OptionInfo(ROPE_DEFAULT, ROPE_LABEL, gr.Slider,
                          {"minimum": rope.MIN_SCALE, "maximum": rope.MAX_SCALE, "step": 0.01}, section=SECTION)
        info = info.info("as ComfyUI's ScaleROPE: below 1 squeezes the image positions, so pictures well above 1 MP "
                         "keep the framing of 1 MP instead of turning into a wider scene with smaller subjects; 0.75 "
                         "to 0.8 suits 2 to 4 MP, lower smears detail ; 1 changes nothing ; ignored when Auto is on")
        shared.opts.add_option(ROPE_OPTION, info)
    if ROPE_AUTO_OPTION not in shared.opts.data_labels:
        info = OptionInfo(ROPE_AUTO_DEFAULT, ROPE_AUTO_LABEL, section=SECTION)
        info = info.info("on by default: sets the scale from each pass's area (Hires fix too): 1 up to 1 MP, then the "
                         "square root of 1 MP over the area, never below 0.8 (0.82 at 1.5 MP, 0.8 from 1.56 MP)")
        shared.opts.add_option(ROPE_AUTO_OPTION, info)
