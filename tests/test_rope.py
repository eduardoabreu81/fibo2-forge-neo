"""The [FIBO 2] RoPE scale: Auto's values, the 1.0 no-op, and the DiT under a scale against ComfyUI's rope_options."""

import pytest
import torch

from forge_fibo2.native import rope, settings
from tests.test_transformer import inputs, models


@pytest.mark.parametrize("size, expected", [
    ((1024, 1024), 1.0),
    ((1216, 832), 1.0),
    ((1248, 1248), 1.0),    # 6084 tokens, under 1.5 MP
    ((1536, 1536), 0.82),
    ((2048, 2048), 0.61),
    ((1792, 1024), 0.93),   # 112 x 64 = 7168 tokens
])
def test_auto_scale(size, expected):
    assert rope.auto_scale(*size) == expected


def test_auto_scale_stops_at_the_slider_minimum():
    assert rope.auto_scale(8192, 8192) == rope.MIN_SCALE


def test_auto_scale_is_by_area_not_by_side():
    # the same area in any proportions gets the same scale
    assert rope.auto_scale(1536, 1536) == rope.auto_scale(2304, 1024) == rope.auto_scale(1024, 2304)


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
    # no modules.shared in the CPU tests: Auto off, slider at 1
    assert settings.rope_scale(2048, 2048) == 1.0


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
