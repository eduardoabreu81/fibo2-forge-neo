# FIBO 2 support for Forge Neo
# The FIBO 2.0 diffusion engine: text-to-image (Dev, and Edit Turbo without an image), image editing with Edit
# Turbo, and plain image-to-image.
#
# Follows Forge Neo's own Krea 2 engine (backend/diffusion_engine/krea.py) for the images to edit: with "[FIBO 2]
# Enable Reference" on, encode_first_stage keeps the img2img image (and ImageStitch's) as pixels; the positive
# prompt's conditioning brings them to upstream's sizes, shows them to Qwen3-VL and encodes them as the DiT's
# reference latents (dynamic_args.ref_latents), which the negative pass uses too.

import contextlib

import torch
from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from . import caption, lora, masking, settings
from .model import TEXT_ENCODER, Fibo2
from .shift import dynamic_shift as shift_of
from .text_engine import Fibo2TextEngine

# the loader's model flags, set from substrings of the checkpoint path and of the configuration folder's path; none
# of them is FIBO 2, and "klein" would swap the Automatic schedule for FLUX.2's
LOADER_FLAGS = ("kontext", "edit", "qwen21", "klein", "wan", "pid", "anima", "krea2", "moss")


@contextlib.contextmanager
def posterior_mean():
    """While open, Forge's VAE encodes to the mean of its posterior instead of a sample of it. Without the class it
    patches (another Forge Neo), encoding samples as before."""
    try:
        from backend.nn import vae as forge_vae
    except ImportError:
        forge_vae = None
    distribution = getattr(forge_vae, "DiagonalGaussianDistribution", None)
    if distribution is None or not hasattr(distribution, "mode"):
        yield
        return
    sample = distribution.sample
    distribution.sample = distribution.mode
    try:
        yield
    finally:
        distribution.sample = sample


class Fibo2Engine(ForgeDiffusionEngine):
    matched_guesses = [Fibo2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={TEXT_ENCODER: huggingface_components["text_encoder"]},
                    tokenizer_dict={TEXT_ENCODER: huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._get_predictor(), config=estimated_config)

        self.text_engine_fibo2 = Fibo2TextEngine(text_encoder=getattr(clip.cond_stage_model, TEXT_ENCODER),
                                                 tokenizer=getattr(clip.tokenizer, TEXT_ENCODER))

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        for flag in LOADER_FLAGS:
            if hasattr(dynamic_args, flag):
                setattr(dynamic_args, flag, False)
        self.is_fibo2 = True
        # the result's (width, height), set by the extension's script (scripts/fibo2.py) as each generation starts:
        # a single image to edit is cropped to it, as upstream does, or with fibo2_fill fitted inside it on grey
        self.fibo2_output_size = None
        self.fibo2_fill = False
        # Forge's LoRA extension is imported by now, whatever the extensions' load order was
        lora.ensure_installed()

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if prompt.is_negative_prompt:
            # text only; an edit's reference latents condition it all the same
            return self.text_engine_fibo2(prompt)

        images = []
        if settings.references_enabled():
            images = [*self.ref_latents]
            if self.ini_latent is not None:
                images.insert(0, self.ini_latent)
                self.ini_latent = None
        if not images:
            dynamic_args.ref_latents.clear()
            return self.text_engine_fibo2(prompt)

        images = caption.edit_images(images, *(self.fibo2_output_size or (None, None)), fill=self.fibo2_fill)
        mask = masking.current()
        if mask is not None and len(images) == 1:
            # a masked edit: as upstream, of one picture only, its masked region greyed out for Qwen3-VL and the VAE
            images = [masking.grey_out(images[0], mask)]
        conds = self.text_engine_fibo2(prompt, images=images)
        dynamic_args.ref_latents = [self.encode_reference(image) for image in images]
        return conds

    @torch.inference_mode()
    def encode_reference(self, image: torch.Tensor) -> torch.Tensor:
        """(1, H, W, 3) in [0, 1] -> its latent (1, 128, H / 16, W / 16), as Forge encodes an img2img image, but the
        mean of the VAE's posterior, as ComfyUI encodes it: Forge samples it with an unseeded torch.randn, so the same
        picture gave a different edit depending on what ran before."""
        x = image.movedim(-1, 1).mul(2.0).sub(1.0)
        with posterior_mean():
            return ForgeDiffusionEngine.encode_first_stage(self, x).cpu()

    def set_shift(self, shift: float = None, width: int = 1024, height: int = 1024, *args, **kwargs):
        # Forge calls this before every sampling pass (hires fix too) with its size; Forge's Qwen-Image 2.1 engine
        # sets its shift from the size the same way. No Shift slider: FIBO 2's shift is the reference pipeline's,
        # from the size (shift.py)
        mu = shift_of(width, height)
        self.forge_objects.unet.model.predictor.set_parameters(shift=mu)
        memory_management.logger.debug(f"[FIBO 2] shift {mu:.4f} for {width}x{height}")

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = self.text_engine_fibo2.tokenize(prompt)
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if settings.references_enabled():
            # the image as pixels: Qwen3-VL sees it, and its latent is made at the size upstream gives it
            image = x[:1, :3].float().movedim(1, -1).mul(0.5).add(0.5).clamp(0.0, 1.0).cpu()
            if dynamic_args.is_referencing:
                self.ref_latents.append(image)
            else:
                self.ini_latent = image

        return super().encode_first_stage(x)
