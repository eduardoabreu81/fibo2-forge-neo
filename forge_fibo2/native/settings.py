# FIBO 2 support for Forge Neo
# "[FIBO 2] Enable Reference", next to Forge Neo's own "[Klein] / [Qwen 2.1] / [Krea2] Enable Reference": on, the
# img2img image (and the ones ImageStitch adds) are images to edit, and the prompt is the edit instruction; off,
# img2img is plain image-to-image.
# "[FIBO 2] Reference Sets the Size": on, a txt2img edit of a single ImageStitch picture comes out about 1 MP in the
# picture's proportions, as Wan2GP's default for Edit Turbo; off, at the size of txt2img's sliders.

OPTION = "fibo2_do_reference"
DEFAULT = True
LABEL = "[FIBO 2] Enable Reference"
SIZE_OPTION = "fibo2_reference_size"
SIZE_DEFAULT = True
SIZE_LABEL = "[FIBO 2] Reference Sets the Size"
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


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        info = OptionInfo(DEFAULT, LABEL, section=SECTION, category_id=CATEGORY)
        info = info.info("enable Edit ; disable img2img").info("pin to <b>Quicksettings</b> is recommended if changed often")
        shared.opts.add_option(OPTION, info)
    if SIZE_OPTION not in shared.opts.data_labels:
        info = OptionInfo(SIZE_DEFAULT, SIZE_LABEL, section=SECTION, category_id=CATEGORY)
        info = info.info("txt2img with one ImageStitch picture: enable about 1 MP in its proportions ; disable the sliders' size")
        shared.opts.add_option(SIZE_OPTION, info)
