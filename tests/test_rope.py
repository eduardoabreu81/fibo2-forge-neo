"""The [FIBO 2] RoPE scale: Auto's values, the 1.0 no-op, and the DiT under a scale against ComfyUI's rope_options."""

import pytest
import torch

from forge_fibo2.native import rope, settings
from tests.test_transformer import inputs, models


@pytest.mark.parametrize("size, expected", [
    ((1024, 1024), 1.0),
    ((1216, 832), 1.0),     # 3952 tokens, under 1 MP
    ((1152, 1152), 0.89),   # 5184 tokens
    ((1248, 1248), 0.82),   # 6084 tokens, about 1.5 MP
    ((1280, 1728), 0.8),    # 2.2 MP: the square root (0.69) is under the floor
    ((1536, 1536), 0.8),
    ((1728, 2304), 0.8),    # 4 MP
    ((2048, 2048), 0.8),
])
def test_auto_scale(size, expected):
    assert rope.auto_scale(*size) == expected


def test_auto_scale_never_goes_below_its_floor():
    assert rope.auto_scale(8192, 8192) == rope.AUTO_FLOOR


def test_auto_scale_is_by_area_not_by_side():
    # the same area in any proportions gets the same scale
    assert rope.auto_scale(1152, 1152) == rope.auto_scale(1728, 768) == rope.auto_scale(768, 1728) == 0.89


def test_slider_values_below_the_minimum_are_raised():
    # a config.json from before the 0.7 minimum may hold 0.5
    assert rope.clamp(0.5) == rope.MIN_SCALE == 0.7
    assert rope.clamp(1.2) == 1.0


def test_one_leaves_the_options_alone():
    options = {"cond_or_uncond": [0]}
    assert rope.with_scale(options, 1.0) is options
    assert rope.with_scale(None, 1.0) is None


def test_scale_becomes_rope_options_without_touching_the_caller():
    options = {"cond_or_uncond": [0]}
    scaled = rope.with_scale(options, 0.8)
    assert scaled["rope_options"] == {"scale_x": 0.8, "scale_y": 0.8}
    assert scaled["cond_or_uncond"] == [0]
    assert "rope_options" not in options


def test_rope_options_set_elsewhere_win():
    options = {"rope_options": {"scale_x": 0.7, "scale_y": 0.9}}
    assert rope.with_scale(options, 0.8) is options


def test_settings_outside_forge_are_the_defaults():
    # no modules.shared in the CPU tests: Auto on, so 0.8 at 4 MP and 1 at 1 MP
    assert settings.rope_scale(2048, 2048) == 0.8
    assert settings.rope_scale(1024, 1024) == 1.0


@torch.no_grad()
def test_dit_at_scale_one_is_unchanged():
    _, ours = models()
    x, context = inputs()
    sigma = torch.tensor([0.7, 0.7])
    before = ours.run(x, sigma, context)
    ours.rope_scale = 1.0
    assert torch.equal(ours(x, sigma, context, ref_latents=[]), before)


@torch.no_grad()
def test_dit_scale_matches_comfyui_rope_options():
    _, ours = models()
    x, context = inputs()
    sigma = torch.tensor([0.7, 0.7])
    expected = ours(x, sigma, context, transformer_options={"rope_options": {"scale_x": 0.8, "scale_y": 0.8}},
                    ref_latents=[])
    ours.rope_scale = 0.8
    scaled = ours(x, sigma, context, ref_latents=[])
    assert torch.equal(scaled, expected)
    assert not torch.equal(scaled, ours.run(x, sigma, context))
