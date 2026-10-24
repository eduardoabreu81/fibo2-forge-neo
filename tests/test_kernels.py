"""The fused comfy-kitchen paths of kernels.py against the eager code they replace (docs/fused-kernels-plan.md).

Runs on comfy-kitchen's eager (CPU) backend, which the kernels fall back to off CUDA; skipped where comfy-kitchen is
not installed. The gate itself is tested with stand-ins, since the CPU has no CUDA tensors.
"""

import types
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from forge_fibo2.native import kernels, transformer

pytest.importorskip("comfy_kitchen")

EPS = 1e-5
TOL = dict(atol=1e-5, rtol=1e-5)


def norm(dim, seed=0):
    result = transformer.RMSNorm(dim, EPS)
    with torch.no_grad():
        result.weight.copy_(torch.randn(dim, generator=torch.Generator().manual_seed(seed)) * 0.3 + 1.0)
    return result


def randn(*shape, seed=1):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def test_modulated_rms_norm_folds_the_weight_into_the_scale():
    n = norm(64)
    x, scale = randn(2, 10, 64), randn(2, 1, 64, seed=2) * 0.5
    expected = n(x) * (1.0 + scale)
    torch.testing.assert_close(kernels.modulated_rms_norm(n, x, scale), expected, **TOL)


@torch.no_grad()
def test_rms_norm_is_the_affine_norm():
    n = norm(64)
    x = randn(2, 10, 64)
    torch.testing.assert_close(kernels.rms_norm(n, x), n(x), **TOL)
    # per head: the last dim is the normalized one
    head_norm = norm(16, seed=3)
    q = randn(2, 10, 4, 16, seed=4)
    torch.testing.assert_close(kernels.rms_norm(head_norm, q), head_norm(q), **TOL)


@torch.no_grad()
def test_fold_scale_is_one_kernel_of_the_same_value():
    n = norm(64)
    scale = randn(2, 1, 64, seed=2) * 0.5
    expected = n.weight * (1.0 + scale) - 1.0
    torch.testing.assert_close(kernels.fold_scale(n, scale, "cpu"), expected, **TOL)
    # the (B, 1, D) view of a (B, 4, D) modulation, as the block hands it over
    mod = randn(2, 4, 64, seed=3) * 0.5
    out = kernels.fold_scale(n, mod[:, 2:3], "cpu")
    assert out.is_contiguous() and out.shape == (2, 1, 64)
    torch.testing.assert_close(out, n.weight * (1.0 + mod[:, 2:3]) - 1.0, **TOL)


@torch.no_grad()
def test_constants_follow_the_weight():
    n = norm(64)
    first = kernels.affine_scale(n, torch.float32, "cpu")
    torch.testing.assert_close(first, n.weight - 1.0, **TOL)
    assert kernels.affine_scale(n, torch.float32, "cpu") is first  # cached
    assert kernels.affine_scale(n, torch.float64, "cpu") is not first  # per dtype
    # written in place (Forge baking a LoRA into the weight): recomputed
    n.weight.mul_(2.0)
    second = kernels.affine_scale(n, torch.float32, "cpu")
    assert second is not first
    torch.testing.assert_close(second, n.weight - 1.0, **TOL)
    # replaced: recomputed
    n.weight = torch.nn.Parameter(torch.full((64,), 1.5))
    torch.testing.assert_close(kernels.affine_scale(n, torch.float32, "cpu"), torch.full((64,), 0.5), **TOL)
    # the fold's w - 1 is kept the same way, and follows a replaced weight too
    scale = randn(2, 1, 64, seed=2) * 0.5
    torch.testing.assert_close(kernels.fold_scale(n, scale, "cpu"), 1.5 * (1.0 + scale) - 1.0, **TOL)
    n.weight = torch.nn.Parameter(torch.full((64,), 0.5))
    torch.testing.assert_close(kernels.fold_scale(n, scale, "cpu"), 0.5 * (1.0 + scale) - 1.0, **TOL)
    # a state dict load writes in place as well
    n.load_state_dict({"weight": torch.full((64,), 2.0)})
    torch.testing.assert_close(kernels.affine_scale(n, torch.float32, "cpu"), torch.full((64,), 1.0), **TOL)


@torch.no_grad()
def test_fused_norms_apply_forge_on_the_fly_patches():
    n = norm(64)
    x, scale = randn(2, 10, 64), randn(2, 1, 64, seed=2) * 0.5
    plain = kernels.rms_norm(n, x)
    n.weight_function.append(lambda w: w * 2.0)
    # a patch bypasses the cached constants: the same weight tensor, another result
    torch.testing.assert_close(kernels.rms_norm(n, x), n(x), **TOL)
    torch.testing.assert_close(kernels.rms_norm(n, x), plain * 2.0, **TOL)
    torch.testing.assert_close(kernels.modulated_rms_norm(n, x, scale), n(x) * (1.0 + scale), **TOL)
    n.weight_function.clear()
    torch.testing.assert_close(kernels.rms_norm(n, x), plain, **TOL)


@torch.no_grad()
def test_norm_rope_matches_the_eager_norm_and_rope():
    heads, head_dim = 2, 128  # the rope axes (32 + 48 + 48) need a 128 head
    ids = torch.cat([transformer.gist_ids(3, "cpu"), transformer.image_ids(513, 2, 3, "cpu")])
    freqs = transformer.rope_embed(ids)
    q, k = randn(2, ids.shape[0], heads, head_dim), randn(2, ids.shape[0], heads, head_dim, seed=5)
    q_norm, k_norm = norm(head_dim, seed=6), norm(head_dim, seed=7)
    expected_q = transformer.apply_rope(q_norm(q.transpose(1, 2)), freqs).transpose(1, 2)
    expected_k = transformer.apply_rope(k_norm(k.transpose(1, 2)), freqs).transpose(1, 2)
    fused_q, fused_k = kernels.norm_rope(q, k, q_norm, k_norm, kernels.rope_freqs(freqs))
    torch.testing.assert_close(fused_q, expected_q, **TOL)
    torch.testing.assert_close(fused_k, expected_k, **TOL)


@torch.no_grad()
def test_linear_swiglu_outside_forge_is_the_plain_swiglu():
    linear = torch.nn.Linear(6, 4, bias=False)
    x = randn(2, 5, 12)
    gate, up = x.chunk(2, dim=-1)
    torch.testing.assert_close(kernels.linear_swiglu(linear, x), linear(F.silu(gate) * up), **TOL)


@torch.no_grad()
def test_gated_residual_updates_each_span_in_place():
    x, y = randn(2, 9, 8), randn(2, 9, 8, seed=2)
    gates = [randn(2, 1, 8, seed=3), randn(2, 1, 8, seed=4)]
    spans = [(0, 4, gates[0]), (4, 9, gates[1])]
    # the eager _gated takes the raw modulation and applies tanh itself
    mods = [(a, e, (None, m, None, None)) for a, e, m in spans]
    expected = transformer._gated(x.clone(), y, mods, 1)
    out = kernels.gated_residual_(x, y, [(a, e, torch.tanh(g)) for a, e, g in spans])
    assert out is x
    torch.testing.assert_close(out, expected, **TOL)


def test_fused_gate_needs_forge_cuda_inference_and_a_wide_dtype():
    cuda_like = types.SimpleNamespace(is_cuda=True)
    cpu_like = types.SimpleNamespace(is_cuda=False)
    # outside Forge (no backend.attention) nothing is fused, whatever else holds
    with torch.no_grad():
        assert not kernels.fused(cuda_like, torch.bfloat16)
    with mock.patch.object(kernels, "_forge_attention", object()):
        assert kernels.available()
        with torch.no_grad():
            assert kernels.fused(cuda_like, torch.bfloat16)
            assert kernels.fused(cuda_like, torch.float32)
            assert not kernels.fused(cuda_like, torch.float16)  # RTX 20 keeps the eager path
            assert not kernels.fused(cpu_like, torch.bfloat16)
        with torch.enable_grad():
            assert not kernels.fused(cuda_like, torch.bfloat16)
    with mock.patch.object(kernels, "_ck", None):
        assert not kernels.available()


def test_swiglu_fold_reads_the_int8_convrot_layout():
    def linear(layout, convrot=True, groupsize=256, patches=()):
        weight = types.SimpleNamespace(_layout_cls=layout,
                                       _params=types.SimpleNamespace(convrot=convrot, convrot_groupsize=groupsize))
        return types.SimpleNamespace(weight=weight, weight_function=list(patches))

    with mock.patch.object(kernels, "_forge_linear_input_act", lambda *a: None):
        assert kernels.swiglu_fold(linear("TensorWiseINT8Layout"))
        assert not kernels.swiglu_fold(linear("TensorWiseINT8Layout", convrot=False))
        assert not kernels.swiglu_fold(linear("TensorWiseINT8Layout", groupsize=128))
        assert not kernels.swiglu_fold(linear("TensorCoreFP8Layout"))
        # an on-the-fly LoRA makes Forge dequantize the weight: SwiGLU runs eagerly then
        assert not kernels.swiglu_fold(linear("TensorWiseINT8Layout", patches=[lambda w: w]))
        # a plain bf16 weight has no layout
        assert not kernels.swiglu_fold(torch.nn.Linear(4, 4))
    # without linear_input_act (older Forge Neo) there is no fold at all
    assert not kernels.swiglu_fold(linear("TensorWiseINT8Layout"))


def fused_path():
    return mock.patch.object(kernels, "fused", lambda x, dtype: True)


def block(cap_feat_dim=None, seed=0):
    result = transformer.Block(256, 2, 96, True, EPS, cap_feat_dim)
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in result.named_parameters():
            if "norm" in name:
                param.uniform_(0.5, 1.5)
            elif name.endswith("cross_attn_gate"):
                param.fill_(0.7)
            else:
                param.normal_(0.0, 0.05)
    return result


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("refs", [False, True])
@torch.no_grad()
def test_block_fused_matches_eager(cross, refs):
    # 3 gist tokens at t = 0, a 2x3 image at the step's t, and a 2x2 reference at t = 0
    num_gist, n_img, n_ref = 3, 6, 4 if refs else 0
    freqs = transformer.rope_table(321, num_gist, 2, 3, [(2, 2)] if refs else [], "cpu")
    zero, noisy = randn(2, 256, seed=2), randn(2, 256, seed=3)
    spans = [(0, num_gist, zero), (num_gist, num_gist + n_img, noisy)]
    if refs:
        spans.append((num_gist + n_img, num_gist + n_img + n_ref, zero))
    x = randn(2, num_gist + n_img + n_ref, 256, seed=4)
    context = randn(2, 7, 48, seed=5) if cross else None
    layer = block(48 if cross else None)
    expected = layer(x.clone(), freqs, spans, context, torch.float32)
    own = x.clone()
    actual = layer(own, freqs, spans, context, torch.float32, fused=True)
    assert actual is own  # in place on the residual stream
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_block_fused_with_one_span():
    # the refiner without references: one span covering the stream
    freqs = transformer.rope_table(321, 3, 2, 3, [], "cpu")[3:]
    x, noisy = randn(2, 6, 256, seed=4), randn(2, 256, seed=3)
    layer = block()
    expected = layer(x.clone(), freqs, [(0, 6, noisy)], None, torch.float32)
    actual = layer(x.clone(), freqs, [(0, 6, noisy)], None, torch.float32, fused=True)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_dit_fused_matches_eager():
    from tests.test_transformer import CONFIG, inputs, models

    _, ours = models()
    x, context = inputs()
    sigma = torch.tensor([0.7, 0.7])
    generator = torch.Generator().manual_seed(2)
    refs = [torch.randn(1, CONFIG["in_channels"], 4, 6, generator=generator),
            torch.randn(1, CONFIG["in_channels"], 3, 5, generator=generator)]
    for ref_latents in ([], refs):
        expected = ours.run(x, sigma, context, ref_latents)
        with fused_path():
            actual = ours.run(x, sigma, context, ref_latents)
        assert ours.reported_path[0] == "on"
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    # the eager run after it reports the eager path (and its reason)
    ours.run(x, sigma, context)
    assert ours.reported_path == ("off", "outside Forge")


def test_rope_table_is_cached_per_layout():
    first = transformer.rope_table(513, 256, 4, 6, [(3, 5)], "cpu")
    assert transformer.rope_table(513, 256, 4, 6, [(3, 5)], "cpu") is first
    other = transformer.rope_table(513, 256, 4, 6, [], "cpu")
    assert other is not first and other.shape[0] == 256 + 24
    scaled = transformer.rope_table(513, 256, 4, 6, [], "cpu", {"rope_options": {"scale_x": 0.5}})
    assert scaled is not other
    assert not torch.equal(scaled, other)
