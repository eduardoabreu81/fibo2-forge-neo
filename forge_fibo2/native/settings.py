# FIBO 2 support for Forge Neo
# Three options in Forge Neo's "Stable Diffusion" settings section, where its per-model options live.
# "[FIBO 2] Enable Reference", next to Forge Neo's own "[Klein] / [Qwen 2.1] / [Krea2] Enable Reference": on, the
# img2img image (and the ones ImageStitch adds) are images to edit, and the prompt is the edit instruction; off,
# img2img is plain image-to-image.
# "[FIBO 2] Reference Sets the Size": on, a txt2img edit of a single ImageStitch picture comes out about 1 MP in the
# picture's proportions, as Wan2GP's default for Edit Turbo, or, when the sliders ask for other proportions, at
# their size with the sides around the picture filled in; off, at the size of txt2img's sliders, the picture cropped.
# "[FIBO 2] Fused kernels": on, the DiT runs on comfy-kitchen's fused kernels where it can (kernels.py); off, the
# eager path, the same code as before the fused one existed, for A/B pictures and timings and for bug reports.

OPTION = "fibo2_do_reference"
DEFAULT = True
LABEL = "[FIBO 2] Enable Reference"
SIZE_OPTION = "fibo2_reference_size"
SIZE_DEFAULT = True
SIZE_LABEL = "[FIBO 2] Reference Sets the Size"
FUSED_OPTION = "fibo2_fused_kernels"
FUSED_DEFAULT = True
FUSED_LABEL = "[FIBO 2] Fused kernels"
# Forge Neo's "Stable Diffusion" settings section and category, where its per-model options live
SECTION = ("sd", "Stable Diffusion")
CATEGORY = "sd"


def _get(option: str, default: bool) -> bool:
    try:
        from modules.shared import opts

        return bool(getattr(opts, option, default))
    except Exception:  # outside Forge Neo (CPU tests), or before the options exist
        return default


def references_enabled() -> bool:
    return _get(OPTION, DEFAULT)


def reference_sets_size() -> bool:
    return _get(SIZE_OPTION, SIZE_DEFAULT)


def fused_enabled() -> bool:
    return _get(FUSED_OPTION, FUSED_DEFAULT)


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        info = OptionInfo(DEFAULT, LABEL, section=SECTION, category_id=CATEGORY)
        info = info.info("enable Edit ; disable img2img").info("pin to <b>Quicksettings</b> is recommended if changed often")
        shared.opts.add_option(OPTION, info)
    if SIZE_OPTION not in shared.opts.data_labels:
        info = OptionInfo(SIZE_DEFAULT, SIZE_LABEL, section=SECTION, category_id=CATEGORY)
        info = info.info("txt2img with one ImageStitch picture: enable about 1 MP in its proportions, or the sliders' "
                         "size with the sides filled in when theirs differ ; disable the sliders' size, cropped")
        shared.opts.add_option(SIZE_OPTION, info)
    if FUSED_OPTION not in shared.opts.data_labels:
        info = OptionInfo(FUSED_DEFAULT, FUSED_LABEL, section=SECTION, category_id=CATEGORY)
        info = info.info("comfy-kitchen fused norms, RoPE and SwiGLU in the DiT (bf16 and INT8 ConvRot files on CUDA) ; "
                         "off runs the plain path, for comparisons")
        shared.opts.add_option(FUSED_OPTION, info)
