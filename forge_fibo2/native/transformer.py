# FIBO 2 support for Forge Neo
# The FIBO 2.0 DiT (Bria's Fibo 2.0 Dev and Edit Turbo), ported from ComfyUI's comfy/ldm/fibo2/model.py
# (COMFYUI_COMMIT in native/__init__.py), which matches the diffusers reference to 4e-7 in float32.
#
# Z-Image's blocks: attention and a SwiGLU MLP, each between two RMS norms (the second one normalizes the branch's
# output), under a scale and a tanh gate from the time embedding. Two refiner blocks run the image and the reference
# images alone; a Perceiver squeezes the text into 256 to 512 "gist" tokens; then 44 blocks run [gist, image,
# references] together, five of them (6, 14, 22, 30, 38) also cross-attending, through a tanh gate, to a bundle of
# text layers of their own. Gist and reference tokens are modulated at t = 0, the image at the step's t.
#
# Layers are plain torch modules: inside Forge's using_forge_operations the linears become Forge operations (dtype,
# INT8 ConvRot quantization, offload, LoRA), outside it they stay torch modules for CPU tests. Module names are
# ComfyUI's, which the Comfy-Org files and their LoRAs use.
# Differences from ComfyUI: norms and modulation run in float32; under float16 the residual streams stay in float32
# and the output projections run on scaled inputs (see project); reference latents come from Forge's
# dynamic_args.ref_latents; a conditioning may carry two text variants (phases.py).
#
# Two paths through every block (docs/fused-kernels-plan.md): the eager one above, and a fused one on comfy-kitchen's
# kernels (kernels.py) that run decides on once per forward (kernels.fused): modulated norms in one pass, q/k norms and
# RoPE in one, gated residuals in place, SwiGLU folded into the w2 INT8 GEMM. The fused path computes in the model's
# dtype (bf16 or float32) throughout; float16 keeps the eager path.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels, phases

THETA = 256.0
AXES_DIM = (32, 48, 48)
FREQ_DIM = 256
TIME_HIDDEN = 1024
ADALN_DIM = 256
# the text: Qwen3-VL bundles of 3 layers x 2560, one for the Perceiver and one per injection block
TEXT_BUNDLES = 6
OUT_FP16_SCALE = 2.0 ** -6
FINAL_EPS = 1e-6


def timestep_embedding(t, dim=FREQ_DIM, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def gist_count(text_len: int, max_gist: int) -> int:
    # 256 gist tokens up to 256 text tokens, rising linearly to max_gist at 2048, in steps of 64
    ratio = min(1.0, max(0.0, (text_len - 256) / (2048 - 256)))
    return max(256, min(max_gist, math.ceil((256 + ratio * (max_gist - 256)) / 64) * 64))


def rope(pos, dim, theta):
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    out = pos.double()[..., None] * (1.0 / theta ** scale)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_embed(ids):
    """ids (L, 3) -> rotation matrices (L, head_dim // 2, 2, 2)."""
    return torch.cat([rope(ids[:, i], AXES_DIM[i], THETA) for i in range(len(AXES_DIM))], dim=-3)


def apply_rope(x, freqs):
    # x [B, H, L, D] in float32, freqs (L, D // 2, 2, 2)
    b, h, length, d = x.shape
    x_r = x.reshape(b, h, length, d // 2, 1, 2)
    return (freqs[..., 0] * x_r[..., 0] + freqs[..., 1] * x_r[..., 1]).reshape(b, h, length, d)


def gist_ids(count, device):
    # gist tokens on rope planes 1 .. count
    ids = torch.zeros(count, 3, dtype=torch.float32, device=device)
    ids[:, 0] = torch.arange(1, count + 1, dtype=torch.float32, device=device)
    return ids


def image_ids(plane, h, w, device, transformer_options=None):
    # ComfyUI's pos_ids_x, with its rope_options (ScaleROPE) when they are set
    rope_options = (transformer_options or {}).get("rope_options") or {}
    ids = torch.zeros(h, w, 3, dtype=torch.float32, device=device)
    ids[..., 0] = plane
    ids[..., 1] = (torch.arange(h, dtype=torch.float32, device=device) * rope_options.get("scale_y", 1.0)
                   + rope_options.get("shift_y", 0.0))[:, None]
    ids[..., 2] = (torch.arange(w, dtype=torch.float32, device=device) * rope_options.get("scale_x", 1.0)
                   + rope_options.get("shift_x", 0.0))[None, :]
    return ids.flatten(0, 1)


_rope_cache = [None, None]


def rope_table(plane, num_gist, h, w, ref_sizes, device, transformer_options=None):
    """The rotation matrices (L, head_dim // 2, 2, 2) of the gist, image and reference tokens, in that order.

    One entry is kept: every block of every step of a generation at one size reads the same table, and building it
    (float64 trigonometry) is not free.
    """
    rope_options = (transformer_options or {}).get("rope_options") or {}
    key = (plane, num_gist, h, w, tuple(ref_sizes), str(device), repr(rope_options))
    if _rope_cache[0] != key:
        ids = [gist_ids(num_gist, device), image_ids(plane, h, w, device, transformer_options)]
        for k, (ref_h, ref_w) in enumerate(ref_sizes):
            ids.append(image_ids(plane + 1 + k, ref_h, ref_w, device, transformer_options))
        _rope_cache[:] = key, rope_embed(torch.cat(ids))
    return _rope_cache[1]


def project(linear, x, dtype):
    """linear(x) for an output projection without bias. Under float16 it runs on x scaled down by a power of two and
    is scaled back in float32 (exact in floating point), so outputs past 65504 survive: ComfyUI clamps them instead,
    and leaves float16 out of the model's inference dtypes."""
    if dtype == torch.float16:
        return linear(x * OUT_FP16_SCALE).float() / OUT_FP16_SCALE
    return linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        # torch.ones, not empty: the checkpoint overwrites it
        self.weight = nn.Parameter(torch.ones(dim))
        # Forge's on-the-fly LoRA (GGUF models, low VRAM) hangs the patches of a module's weight here, as on its own
        # operations (backend.operations.ForgeWeights), and the module applies them; one list per norm
        self.weight_function = []
        self.bias_function = []

    def forward(self, x):
        """RMS norm in float32, returned in float32."""
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * kernels.norm_weight(self, x.device)


class FeedForward(nn.Module):
    # SwiGLU; the Comfy-Org files carry w1 and w3 as one w13 (gate half first)
    def __init__(self, dim, hidden_dim, fused):
        super().__init__()
        self.fused = fused
        if fused:
            self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def hidden(self, x):
        gate, up = self.w13(x).chunk(2, dim=-1) if self.fused else (self.w1(x), self.w3(x))
        return F.silu(gate) * up

    def forward_fused(self, x):
        """w2(swiglu(x)) in x's dtype; with a w13, SwiGLU folds into w2's INT8 GEMM where Forge can (kernels)."""
        if self.fused:
            return kernels.linear_swiglu(self.w2, self.w13(x))
        return self.w2(self.hidden(x))


class JointAttention(nn.Module):
    # self-attention with RoPE over the whole sequence; q, k, v fused, as ComfyUI's Lumina JointAttention
    def __init__(self, dim, n_heads, eps):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(dim // n_heads, eps)
        self.k_norm = RMSNorm(dim // n_heads, eps)

    def forward(self, x, freqs, transformer_options=None, fused=False):
        """x (B, N, D) -> attention before the out projection, (B, N, D); freqs (N, D // n_heads // 2, 2, 2)."""
        b, n, _ = x.shape
        if fused:
            # (B, N, H, D) views of the projection, the layout of comfy-kitchen's RoPE kernels
            q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, -1).unbind(2)
            q, k = kernels.norm_rope(q, k, self.q_norm, self.k_norm, kernels.rope_freqs(freqs))
            return kernels.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), transformer_options)
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q = apply_rope(self.q_norm(q), freqs).to(x.dtype)
        k = apply_rope(self.k_norm(k), freqs).to(x.dtype)
        return kernels.attention(q, k, v, transformer_options)


class Attention(nn.Module):
    # no RoPE: the cross-attention to the text, and the Perceiver's self-attention
    def __init__(self, dim, n_heads, kv_dim, eps):
        super().__init__()
        self.n_heads = n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(dim // n_heads, eps)
        self.norm_k = RMSNorm(dim // n_heads, eps)

    def forward(self, x, context, transformer_options=None, fused=False):
        """x (B, N, D), context (B, M, kv_dim) -> attention before to_out, (B, N, D)."""
        q = self.to_q(x).unflatten(-1, (self.n_heads, -1))
        k, v = self.to_kv(context).unflatten(-1, (2, self.n_heads, -1)).unbind(2)
        if fused:
            q, k = kernels.rms_norm(self.norm_q, q), kernels.rms_norm(self.norm_k, k)
        else:
            q, k = self.norm_q(q).to(x.dtype), self.norm_k(k).to(x.dtype)
        return kernels.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2).to(x.dtype),
                                 transformer_options)


def _modulations(block, spans):
    """[(start, end, adaln)] -> [(start, end, (scale_msa, gate_msa, scale_mlp, gate_mlp))], each (B, 1, D) float32;
    the adaLN runs once per distinct time embedding."""
    cache = {}
    out = []
    for start, end, adaln in spans:
        if id(adaln) not in cache:
            cache[id(adaln)] = tuple(m.float()[:, None] for m in block.adaLN_modulation(adaln).chunk(4, dim=-1))
        out.append((start, end, cache[id(adaln)]))
    return out


def _modulate(normed, mods, index, dtype):
    return torch.cat([normed[:, a:e] * (1.0 + m[index]) for a, e, m in mods], dim=1).to(dtype)


def _gated(x, y, mods, index):
    return torch.cat([x[:, a:e] + (torch.tanh(m[index]) * y[:, a:e]).to(x.dtype) for a, e, m in mods], dim=1)


def _modulations_fused(block, spans, device):
    """[(start, end, adaln)] -> [(start, end, (scale_msa, gate_msa, scale_mlp, gate_mlp))] for the fused path, each
    (B, 1, D) in the compute dtype, once per distinct time embedding and in four kernels: the adaLN projection, one
    tanh over both gates, and the two scales folded with their norm weights (kernels.fold_scale)."""
    cache = {}
    out = []
    for start, end, adaln in spans:
        if id(adaln) not in cache:
            mod = block.adaLN_modulation(adaln).view(adaln.shape[0], 4, -1)
            gates = torch.tanh(mod[:, 1::2])
            cache[id(adaln)] = (kernels.fold_scale(block.attention_norm1, mod[:, 0:1], device), gates[:, 0:1],
                                kernels.fold_scale(block.ffn_norm1, mod[:, 2:3], device), gates[:, 1:2])
        out.append((start, end, cache[id(adaln)]))
    return out


def _modulated_fused(norm, x, mods, index):
    """rmsnorm(x) * (1 + folded scale) per span in one kernel pass each, in x's dtype (the spans cover x)."""
    if len(mods) == 1:
        return kernels.modulated_fused(x, mods[0][2][index], norm.eps)
    return torch.cat([kernels.modulated_fused(x[:, a:e], m[index], norm.eps) for a, e, m in mods], dim=1)


def _gates(mods, index):
    return [(a, e, m[index]) for a, e, m in mods]


class Block(nn.Module):
    # Z-Image's block; an injection block also cross-attends to its own text bundle
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, eps, cap_feat_dim=None):
        super().__init__()
        self.attention = JointAttention(dim, n_heads, eps)
        self.feed_forward = FeedForward(dim, ffn_dim, fused_ffn)
        self.attention_norm1 = RMSNorm(dim, eps)
        self.ffn_norm1 = RMSNorm(dim, eps)
        self.attention_norm2 = RMSNorm(dim, eps)
        self.ffn_norm2 = RMSNorm(dim, eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(ADALN_DIM, 4 * dim, bias=True))
        self.cross_attention = None
        if cap_feat_dim is not None:
            self.cross_attn_norm1 = RMSNorm(dim, eps)
            self.cross_attention = Attention(dim, n_heads, cap_feat_dim, eps)
            self.cross_attn_norm2 = RMSNorm(dim, eps)
            self.cross_attn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, freqs, spans, context, dtype, transformer_options=None, fused=False):
        """x (B, N, D) residual stream; spans [(start, end, adaln)] cover x, adaln (B, 256) the time embedding that
        modulates those rows; context (B, M, cap_feat_dim) for an injection block. Fused: x is in dtype and is
        updated in place."""
        if fused:
            return self.forward_fused(x, freqs, _modulations_fused(self, spans, x.device), context,
                                      transformer_options)
        mods = _modulations(self, spans)
        y = self.attention(_modulate(self.attention_norm1(x), mods, 0, dtype), freqs, transformer_options)
        x = _gated(x, self.attention_norm2(project(self.attention.out, y, dtype)), mods, 1)
        if self.cross_attention is not None:
            y = self.cross_attention(self.cross_attn_norm1(x).to(dtype), context, transformer_options)
            y = self.cross_attn_norm2(project(self.cross_attention.to_out, y, dtype))
            gate = torch.tanh(kernels.cast_to(self.cross_attn_gate, torch.float32, x.device))
            x = x + (gate * y).to(x.dtype)
        y = self.feed_forward.hidden(_modulate(self.ffn_norm1(x), mods, 2, dtype))
        return _gated(x, self.ffn_norm2(project(self.feed_forward.w2, y, dtype)), mods, 3)

    def forward_fused(self, x, freqs, mods, context, transformer_options=None):
        """The block on comfy-kitchen's kernels: the same arithmetic as forward, in x's dtype, x updated in place;
        mods from _modulations_fused."""
        y = self.attention(_modulated_fused(self.attention_norm1, x, mods, 0), freqs, transformer_options, fused=True)
        x = kernels.gated_residual_(x, kernels.rms_norm(self.attention_norm2, self.attention.out(y)), _gates(mods, 1))
        if self.cross_attention is not None:
            y = self.cross_attention(kernels.rms_norm(self.cross_attn_norm1, x), context, transformer_options,
                                     fused=True)
            y = kernels.rms_norm(self.cross_attn_norm2, self.cross_attention.to_out(y))
            gate = kernels.constant(self, f"_fibo2_gate_{x.dtype}", self.cross_attn_gate,
                                    lambda g: torch.tanh(kernels.cast_to(g, x.dtype, x.device)))
            x.addcmul_(y, gate)
        y = self.feed_forward.forward_fused(_modulated_fused(self.ffn_norm1, x, mods, 2))
        return kernels.gated_residual_(x, kernels.rms_norm(self.ffn_norm2, y), _gates(mods, 3))


class PerceiverLayer(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, eps):
        super().__init__()
        self.cross_attn_norm = RMSNorm(dim, eps)
        self.cross_attention = Attention(dim, n_heads, cap_feat_dim, eps)
        self.self_attn_norm = RMSNorm(dim, eps)
        self.self_attention = Attention(dim, n_heads, dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)
        self.ffn = FeedForward(dim, ffn_dim, fused_ffn)

    def forward(self, x, context, dtype, transformer_options=None, fused=False):
        if fused:
            # out of place: the first layer's x is the expanded latent queries
            y = self.cross_attention(kernels.rms_norm(self.cross_attn_norm, x), context, transformer_options, True)
            x = x + self.cross_attention.to_out(y)
            y = kernels.rms_norm(self.self_attn_norm, x)
            x = x + self.self_attention.to_out(self.self_attention(y, y, transformer_options, True))
            return x + self.ffn.forward_fused(kernels.rms_norm(self.ffn_norm, x))
        y = self.cross_attention(self.cross_attn_norm(x).to(dtype), context, transformer_options)
        x = x + project(self.cross_attention.to_out, y, dtype).to(x.dtype)
        y = self.self_attn_norm(x).to(dtype)
        x = x + project(self.self_attention.to_out, self.self_attention(y, y, transformer_options), dtype).to(x.dtype)
        y = self.ffn.hidden(self.ffn_norm(x).to(dtype))
        return x + project(self.ffn.w2, y, dtype).to(x.dtype)


class Perceiver(nn.Module):
    # squeezes the text encoder's hidden states into gist tokens
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, num_queries, num_layers, eps):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.zeros(num_queries, dim))
        self.layers = nn.ModuleList([PerceiverLayer(dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, eps)
                                     for _ in range(num_layers)])
        self.norm_out = RMSNorm(dim, eps)

    def forward(self, context, num_gist, dtype, res_dtype, transformer_options=None, fused=False):
        """context (B, M, cap_feat_dim) -> gist tokens (B, num_gist, D) in float32 (fused: in res_dtype)."""
        queries = kernels.cast_to(self.latent_queries, dtype, context.device)[:num_gist].to(res_dtype)
        x = queries[None].expand(context.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, context, dtype, transformer_options, fused)
        return kernels.rms_norm(self.norm_out, x) if fused else self.norm_out(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(FREQ_DIM, hidden, bias=True), nn.SiLU(), nn.Linear(hidden, out, bias=True))

    def forward(self, t, dtype):
        return self.mlp(timestep_embedding(t).to(dtype))


class FinalLayer(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(ADALN_DIM, dim, bias=True))

    def forward(self, x, adaln, dtype):
        # LayerNorm without affine, then the scale, in float32; x is the residual stream in its own dtype
        scale = self.adaLN_modulation(adaln).float()[:, None]
        normed = F.layer_norm(x.float(), (x.shape[-1],), eps=FINAL_EPS)
        return self.linear((normed * (1.0 + scale)).to(dtype))


class Fibo2DiT(nn.Module):
    def __init__(self, in_channels=128, dim=3584, n_heads=28, n_layers=44, n_refiner_layers=2,
                 injection_layers=(6, 14, 22, 30, 38), cap_feat_dim=7680, perceiver_layers=2, max_gist_tokens=512,
                 ffn_dim=9728, fused_ffn=True, norm_eps=1e-5, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.max_gist_tokens = max_gist_tokens
        self.injection_layers = tuple(injection_layers)
        self.text_dim = (1 + len(self.injection_layers)) * cap_feat_dim

        self.x_embedder = nn.Linear(in_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(TIME_HIDDEN, ADALN_DIM)
        self.noise_refiner = nn.ModuleList([Block(dim, n_heads, ffn_dim, fused_ffn, norm_eps)
                                            for _ in range(n_refiner_layers)])
        self.perceiver = Perceiver(dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, max_gist_tokens, perceiver_layers,
                                   norm_eps)
        self.layers = nn.ModuleList([Block(dim, n_heads, ffn_dim, fused_ffn, norm_eps,
                                           cap_feat_dim if i in self.injection_layers else None)
                                     for i in range(n_layers)])
        self.final_layer = FinalLayer(dim, in_channels)
        # the path of the last forward, as reported in the console (kernels.report)
        self.reported_path = None

    def forward(self, x, timestep, context, transformer_options=None, ref_latents=None, **kwargs):
        """x (B, 128, H, W) latents, timestep (B,) = sigma, context (B, T, text_dim + 1) from the text engine (or
        (B, T, text_dim)) -> velocity (B, 128, H, W)."""
        if ref_latents is None:
            from backend.args import dynamic_args

            ref_latents = dynamic_args.ref_latents
        refs = [r for r in (ref_latents or []) if r is not None]

        if context.shape[-1] != self.text_dim + 1:
            return self.run(x, timestep, context, refs, transformer_options)
        texts = phases.split(context, float(timestep[0]))
        if all(t.shape == texts[0].shape for t in texts):
            return self.run(x, timestep, torch.stack(texts), refs, transformer_options)
        # samples whose prompts differ in length at this step run one by one (the gist count follows the length)
        return torch.cat([self.run(x[i:i + 1], timestep[i:i + 1], text[None], refs, transformer_options)
                          for i, text in enumerate(texts)])

    def run(self, x, timestep, context, refs=(), transformer_options=None):
        b, _, h, w = x.shape
        device = x.device
        # Forge hands the inputs over in the computation dtype (k_model.apply_model); the weights may be stored in
        # another one (INT8, manual cast), so the context is what says which
        dtype = context.dtype
        # fp16 overflows the residual stream of large DiTs (the Iris and FLUX 3 ports saw it); keep it in float32
        res_dtype = torch.float32 if dtype == torch.float16 else dtype
        fused = kernels.fused(x, dtype)
        self.reported_path = kernels.report(self, x, dtype, fused, self.reported_path)

        # the image and the reference tokens are noisy and clean: zero modulates the gist and the references
        timestep = timestep.to(device=device, dtype=torch.float32)
        zero = self.t_embedder(torch.zeros_like(timestep), dtype)
        noisy = self.t_embedder(timestep * 1000.0, dtype)

        # gist tokens on rope planes 1 .. num_gist, the image on plane max_gist_tokens + 1 whatever the gist count,
        # reference k on the k-th plane after it
        num_gist = gist_count(context.shape[1], self.max_gist_tokens)
        plane = self.max_gist_tokens + 1
        tokens = [x.flatten(2).transpose(1, 2)]
        for ref in refs:
            tokens.append(ref.to(device=device, dtype=x.dtype).flatten(2).transpose(1, 2).expand(b, -1, -1))
        img = self.x_embedder(torch.cat(tokens, dim=1).to(dtype)).to(res_dtype)
        freqs = rope_table(plane, num_gist, h, w, [(r.shape[-2], r.shape[-1]) for r in refs], device,
                           transformer_options)
        del tokens

        n_img = h * w
        spans = [(0, n_img, noisy)] + ([(n_img, img.shape[1], zero)] if refs else [])
        for block in self.noise_refiner:
            img = block(img, freqs[num_gist:], spans, None, dtype, transformer_options, fused)

        # the Perceiver reads the first text bundle, each injection block its own
        bundles = context.to(device=device, dtype=dtype).chunk(1 + len(self.injection_layers), dim=-1)
        gist = self.perceiver(bundles[0], num_gist, dtype, res_dtype, transformer_options, fused).to(res_dtype)
        text = dict(zip(self.injection_layers, bundles[1:]))

        x = torch.cat((gist, img), dim=1)
        del gist, img
        end = num_gist + n_img
        spans = [(0, num_gist, zero), (num_gist, end, noisy)] + ([(end, x.shape[1], zero)] if refs else [])
        for i, block in enumerate(self.layers):
            x = block(x, freqs, spans, text.get(i), dtype, transformer_options, fused)

        out = self.final_layer(x[:, num_gist:end], noisy, dtype)
        return out.transpose(1, 2).reshape(b, self.in_channels, h, w)
