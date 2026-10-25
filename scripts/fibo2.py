# FIBO 2 support for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup, before the UI and before any model loads, which is
# when the model gets registered in the checkpoint list. Everything runs in the module scope; any failure leaves
# Forge Neo unchanged and says so in the console.

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENABLED = False

try:
    from forge_fibo2.native import compat

    _problems = compat.check()
    if _problems:
        compat.report(_problems)
    elif (_native := compat.native_support()) is not None:
        compat.report_native(_native)
    else:
        from forge_fibo2.native import patches

        patches.apply()
        ENABLED = True
except Exception:
    print("[FIBO 2] failed to enable, Forge Neo is unchanged:")
    traceback.print_exc()

if ENABLED:
    from modules import scripts

    from forge_fibo2.native import caption, masking, settings

    def stitched(p):
        """The pictures ImageStitch adds to p, as PIL images; none when it is off."""
        for script in getattr(p.scripts, "alwayson_scripts", None) or ():
            if script.title() == "ImageStitch Integrated":
                try:
                    enable, references, *_ = p.script_args[script.args_from:script.args_to]
                    return script.extract_images(references) if enable and references else []
                except Exception:  # arguments not shaped as its UI makes them
                    return []
        return []

    class Fibo2MaskedEdit(scripts.Script):
        """No UI: hands the result's size and img2img's inpaint mask, as Forge prepared it for the picture, to the
        FIBO 2 engine, which crops the picture to edit to that size and greys the masked region out (masking.py);
        says in the console when the size does not suit an edit. In txt2img, a single ImageStitch picture sets the
        result's size, or, where the sliders' proportions are other than its own, is fitted inside their size with
        the sides masked for the edit to fill in; img2img's "Resize and fill" masks the sides it fills the same
        way."""

        def title(self):
            return "FIBO 2 masked edit"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def process(self, p, *args):
            # every generation starts without a mask; img2img's inpaint sets one below, during p.init
            masking.set_mask(None)
            if getattr(p.sd_model, "is_fibo2", False):
                p.sd_model.fibo2_fill = False
                if not self.is_img2img and settings.references_enabled() and settings.reference_sets_size():
                    self.size_from_reference(p)
                p.sd_model.fibo2_output_size = (p.width, p.height)

        @staticmethod
        def size_from_reference(p):
            # as Wan2GP's default for Edit Turbo: a single picture to edit sets the result's size, about 1 MP in its
            # proportions, where the sliders' size would crop and scale it (caption.edit_images). This runs before
            # ImageStitch's process (sorting_priority 0 against its 529), so the picture comes from its arguments
            pictures = stitched(p)
            if len(pictures) != 1:
                return
            picture, size = pictures[0].size, (p.width, p.height)
            if not caption.same_proportions(picture, size):
                # other proportions, as a portrait into 16:9: the sliders' size, the picture fitted inside it and
                # the sides filled in, as Bria's outpainting template does (a size of the picture's own would undo
                # the change the sliders ask for, and a crop would cut the picture down)
                p.sd_model.fibo2_fill = True
                size = (max(16, p.width // 16 * 16), max(16, p.height // 16 * 16))
                masking.set_mask(masking.fill_mask(size, caption.fill_box(picture, size)))
                print(f"[FIBO 2] the {picture[0]}x{picture[1]} ImageStitch picture is fitted inside "
                      f"{p.width}x{p.height} and the sides are filled in; for the picture's own proportions, set "
                      f"the sliders to them")
                return
            better = caption.size_advice(picture, size)
            if better is not None:
                print(f"[FIBO 2] txt2img size set to {better[0]}x{better[1]} from {p.width}x{p.height}, in the "
                      f"proportions of the {picture[0]}x{picture[1]} ImageStitch picture; "
                      f"turn this off under Settings -> Stable Diffusion -> {settings.SIZE_LABEL}")
                p.width, p.height = better

        def before_process_init_images(self, p, pp, *args, **kwargs):
            if not getattr(p.sd_model, "is_fibo2", False):
                return
            masking.set_mask(pp.get("image_mask"))
            if settings.references_enabled() and p.init_images:
                region = pp.get("crop_region")
                picture = (region[2] - region[0], region[3] - region[1]) if region else p.init_images[0].size
                if p.resize_mode == 2 and region is None:
                    box = caption.fill_box(picture, (p.width, p.height))
                    if box != (0, 0, p.width, p.height):
                        self.fill_sides(p, pp.get("image_mask"), masking.fill_mask((p.width, p.height), box))
                        # the picture now has the result's proportions: only the size is left to advise on
                        picture = (p.width, p.height)
                self.advise(picture, (p.width, p.height))

        @staticmethod
        def fill_sides(p, mask, sides):
            # "Resize and fill" stretches the picture's edges out to the result's proportions, which an edit would
            # keep; masked, they are greyed out and filled in instead. Forge has made its latent mask and overlay
            # from the inpaint mask already, so the sides join both: the latent mask is this same picture, drawn
            # on in place, and the overlay would paste the stretched edges back over the result
            if mask is None:
                masking.set_mask(sides)
                return
            from PIL import Image, ImageChops

            mask.paste(Image.new(mask.mode, mask.size, "white"), (0, 0), sides)
            if p.mask_for_overlay is not None:
                overlay = p.mask_for_overlay.convert("L")
                p.mask_for_overlay = ImageChops.lighter(overlay, sides.resize(overlay.size))

        @staticmethod
        def advise(picture, size):
            from modules.shared import opts

            better = caption.size_advice(picture, size, int(getattr(opts, "res_step", 64)))
            if better is not None:
                print(f"[FIBO 2] img2img is set to {size[0]}x{size[1]} for a {picture[0]}x{picture[1]} picture; "
                      f"edits work best at about 1 megapixel in the picture's proportions, such as "
                      f"{better[0]}x{better[1]} with Crop and resize - other sizes can zoom in or lose detail")
