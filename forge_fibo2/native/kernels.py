# FIBO 2 support for Forge Neo
# The few Forge Neo runtime pieces the DiT calls, with plain torch fallbacks outside Forge (CPU tests), and the fused
# comfy-kitchen paths of the DiT block (docs/fused-kernels-plan.md): the same fusions ComfyUI ships for FIBO 2.0
# (Lumina's rms_rope, SwiGLU folded into the w2 INT8 GEMM) and Forge Neo's MOSS-Image carries (modulated RMSNorm as
# one rms_adaln pass, gated residuals as one addcmul).
#
# The fused path runs inside Forge, on CUDA, during inference, with a bf16 or float32 compute dtype: float16 (RTX 20
# series) keeps the eager path, whose float32 residual stream and scaled output projections (transformer.project) the
# fused kernels have no room for. Fibo2DiT.run decides once per forward (fused) and hands the flag to the blocks.
#
# Weight formats: INT8 ConvRot safetensors get everything, the SwiGLU fold included; bf16 safetensors run SwiGLU
# eagerly inside Forge's linear_input_act and get the rest; GGUF loads as before and is not tuned for.

import torch
import torch.nn.functional as F

try:
    from backend import attention as _forge_attention
    from backend.memory_management import cast_to as _forge_cast_to
except ImportError:  # outside Forge Neo: eager backend on CPU
    _forge_attention = None
    _forge_cast_to = None

try:
    # operations first: operations_mixed_precision is half of it and imports it back
    import backend.operations  # noqa: F401
    from backend.operations_mixed_precision import linear_input_act as _forge_linear_input_act
except ImportError:  # outside Forge Neo, or a Forge Neo without fused input activations
    _forge_linear_input_act = None

try:
    import comfy_kitchen as _ck
except ImportError:
    _ck = None

__all__ = ["affine_scale", "attention", "available", "cast_to", "constant", "fold_scale", "fused", "gated_residual_",
           "linear_swiglu", "modulated_fused", "modulated_rms_norm", "norm_rope", "norm_weight", "report", "rms_norm",
           "rope_freqs", "swiglu_fold", "why_not"]

# the compute dtypes of the fused path; float16 keeps the eager one
FUSED_DTYPES = (torch.bfloat16, torch.float32)
_KERNELS = ("rms_adaln", "rms_rope")


def cast_to(weight, dtype=None, device=None):
    if _forge_cast_to is not None:
        return _forge_cast_to(weight, dtype=dtype, device=device)
    return weight.to(dtype=dtype, device=device)


def available() -> bool:
    """Inside Forge, with a comfy-kitchen that has the kernels the fused path calls."""
    return _forge_attention is not None and _ck is not None and all(hasattr(_ck, k) for k in _KERNELS)


def why_not(x, dtype) -> str | None:
    """Why a forward on x, computing in dtype, takes the eager path; None when it takes the fused one. The kernels
    have no autograd (inference only) and the fold needs the stream and the branches in one dtype (not float16, see
    the module comment)."""
    if _forge_attention is None:
        return "outside Forge"
    if not available():
        return "comfy-kitchen with rms_adaln and rms_rope is not installed"
    if not x.is_cuda:
        return "not on CUDA"
    if torch.is_grad_enabled():
        return "gradients are enabled"
    if dtype not in FUSED_DTYPES:
        return f"{str(dtype).removeprefix('torch.')} compute keeps the eager path"
    if not _setting_on():
        return "the [FIBO 2] Fused kernels setting is off"
    return None


def fused(x, dtype) -> bool:
    """Whether a forward on x, computing in dtype, takes the fused path (why_not)."""
    return why_not(x, dtype) is None


def _setting_on() -> bool:
    from . import settings

    return settings.fused_enabled()


def report(model, x, dtype, is_fused, previous):
    """One console line each time the DiT's path changes: fused or eager, and why, and whether the SwiGLU fold
    engages on this model's weights. Returns the state to hand back on the next forward. Quiet outside Forge."""
    if is_fused:
        fold = swiglu_fold(model.layers[0].feed_forward.w2)
        state = ("on", str(dtype).removeprefix("torch."), fold)
        line = (f"fused comfy-kitchen kernels: on ({state[1]}), SwiGLU folded into the INT8 ConvRot GEMM: "
                f"{'yes' if fold else 'no (not an INT8 ConvRot weight)'}")
    else:
        state = ("off", why_not(x, dtype))
        line = f"fused comfy-kitchen kernels: off ({state[1]})"
    if state != previous and _forge_attention is not None:
        print(f"[FIBO 2] {line}")
    return state


def swiglu_fold(linear) -> bool:
    """Whether Forge folds SwiGLU into this linear's GEMM: an INT8 ConvRot weight under linear_input_act. Other
    weights (bf16, GGUF, a weight under an on-the-fly LoRA) run SwiGLU eagerly inside the same call."""
    weight = getattr(linear, "weight", None)
    params = getattr(weight, "_params", None)
    return (_forge_linear_input_act is not None and params is not None
            and getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout" and getattr(params, "convrot", False)
            and getattr(params, "convrot_groupsize", 256) == 256 and len(getattr(linear, "weight_function", ())) == 0)


def norm_weight(norm, device, dtype=torch.float32):
    """A norm's weight with Forge's on-the-fly LoRA patches applied (the module's weight_function list), in dtype."""
    weight = cast_to(norm.weight, dtype, device)
    for patch in norm.weight_function:
        weight = patch(weight)
    return weight


_zeros = {}


def _shift(d, dtype, device):
    # rms_adaln wants a shift; Z-Image's modulation has none. One zeros vector per (D, dtype, device)
    key = (d, dtype, device)
    zeros = _zeros.get(key)
    if zeros is None:
        zeros = _zeros[key] = torch.zeros(d, dtype=dtype, device=device)
    return zeros


def _key(weight):
    # what can change a weight under a module: replaced (Forge's LoRA patching), moved or cast (offload), or
    # written in place (_version counts in-place writes, inside inference mode too)
    return (weight.data_ptr(), str(weight.device), weight.dtype, weight._version)


def constant(owner, name, weight, compute):
    """compute(weight), kept on owner under name until weight is replaced, moved, cast or written in place.

    Step-independent vectors of a block (w - 1 of a norm, a tanh gate) are a kernel launch each per block per step
    when recomputed, and nothing when kept: ~1500 launches less per forward of the 44-block model.
    """
    key = _key(weight)
    entry = owner.__dict__.get(name)
    if entry is None or entry[0] != key:
        entry = (key, compute(weight))
        owner.__dict__[name] = entry
    return entry[1]


def affine_scale(norm, dtype, device):
    """w - 1 in dtype: the norm's weight as an rms_adaln scale. Cached unless an on-the-fly LoRA patches the weight."""
    if norm.weight_function:
        return (norm_weight(norm, device) - 1.0).to(dtype)
    return constant(norm, f"_fibo2_affine_{dtype}", norm.weight,
                    lambda w: (cast_to(w, torch.float32, device) - 1.0).to(dtype))


def fold_scale(norm, scale, device):
    """w * (1 + s) - 1 in s's dtype, one kernel: addcmul(w - 1, w, s) with the cached w - 1. s (B, 1, D).

    The norm weight folds into the modulation: rmsnorm(x) * w * (1 + s) = rmsnorm(x) * (1 + (w * (1 + s) - 1)).
    The sum runs in float32 (the elementwise kernel's working precision) and rounds once on the way out.
    """
    if norm.weight_function:
        return (norm_weight(norm, device) * (1.0 + scale.float()) - 1.0).to(scale.dtype)
    minus_one = constant(norm, "_fibo2_fold", norm.weight, lambda w: cast_to(w, torch.float32, device) - 1.0)
    weight = cast_to(norm.weight, torch.float32, device)
    return torch.addcmul(minus_one, weight, scale, out=torch.empty(scale.shape, dtype=scale.dtype, device=device))


def modulated_fused(x, folded_scale, eps):
    """rmsnorm(x) * (1 + folded_scale) in one pass, in x's dtype; folded_scale from fold_scale."""
    return _ck.rms_adaln(x, folded_scale, _shift(x.shape[-1], x.dtype, x.device), eps)


def modulated_rms_norm(norm, x, scale):
    """norm(x) * (1 + scale) in one pass, in x's dtype. x (B, N, D), scale (B, 1, D)."""
    return modulated_fused(x, fold_scale(norm, scale.to(x.dtype), x.device), norm.eps)


def rms_norm(norm, x):
    """norm(x) in one pass, in x's dtype: rms_adaln with the affine weight as its scale. The last dim is the
    normalized one, so (B, N, D) and a per-head (B, N, H, D) both work."""
    return _ck.rms_adaln(x, affine_scale(norm, x.dtype, x.device), _shift(x.shape[-1], x.dtype, x.device), norm.eps)


def rope_freqs(freqs):
    """The DiT's (L, D / 2, 2, 2) rotation matrices in the layout of comfy-kitchen's RoPE kernels, (1, L, 1, D / 2, 2, 2)."""
    return freqs[None, :, None]


def norm_rope(q, k, q_norm, k_norm, freqs):
    """q, k (B, L, H, D) -> per-head RMSNorm, then interleaved RoPE, in one pass; freqs from rope_freqs.

    FIBO's rope pairs adjacent elements (transformer.apply_rope), so this is comfy-kitchen's interleaved kernel, the
    one ComfyUI's Lumina JointAttention (FIBO 2's parent class) and Forge Neo's own Lumina call.
    """
    q_weight = norm_weight(q_norm, q.device, q.dtype)
    k_weight = norm_weight(k_norm, k.device, k.dtype)
    return _ck.rms_rope(q, k, freqs, q_weight, k_weight, q_norm.eps)


def linear_swiglu(linear, x):
    """linear(silu(gate) * up) for x = [gate | up] (the w13 layout).

    Inside Forge the call is ComfyUI's: linear_input_act folds SwiGLU into comfy-kitchen's INT8 ConvRot activation
    quantization when the weight allows (swiglu_fold), and runs it eagerly before the linear otherwise. Outside Forge
    the plain chunk, silu, mul and linear.
    """
    if _forge_linear_input_act is not None:
        return _forge_linear_input_act(linear, x, "swiglu")
    gate, up = x.chunk(2, dim=-1)
    return linear(F.silu(gate) * up)


def gated_residual_(x, y, gates):
    """x[:, a:e] += gate * y[:, a:e] in place, one addcmul per span; gates [(start, end, gate (B, 1, D))].

    In place: x is the DiT's own residual stream (transformer.Fibo2DiT.run allocates it), never the sampler's tensor.
    """
    for start, end, gate in gates:
        x[:, start:end].addcmul_(y[:, start:end], gate.to(x.dtype))
    return x


def attention(q, k, v, transformer_options=None):
    """q [B, H, Sq, D], k and v [B, H, Sk, D] -> [B, Sq, H * D], no mask.

    transformer_options reach Forge's attention wrapper, which applies attention overrides such as the Sparse
    Attention Integrated extension. The torch fallback (CPU tests) pins the MATH backend, the plain reference
    implementation.
    """
    if _forge_attention is not None:
        # looked up per call: Forge picks its attention function at import time and other extensions may wrap it
        return _forge_attention.attention_function(q, k, v, q.shape[1], skip_reshape=True,
                                                   transformer_options=transformer_options or {})
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
