# src/LexaLCM/LCM_Model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import autocast
from transformers import PreTrainedModel, MODEL_MAPPING
from src.LexaLCM.Config.LCM_Config import LexaLCMConfig
from LexaLCM.Utils.InspectEmbeddings import inspect_embeddings as inspect_util


# ToDo: make these global variables that can be set to True/False from the command line, maybe
Verbose_Model = False
Verbose_Contextualizer = False
Verbose_Denoiser = False
Verbose_Stats_Modulator = False
Verbose_Stats_ModulatorClamping = False
Verbose_Loss = False

## ------------------------------------------------------------
## Helper Layers
## ------------------------------------------------------------

class NormalizeInput(nn.Module): # ToDo: add input normalization as per the Meta FAIR paper
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        return x

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # starts as float32

    def forward(self, x):

        if Verbose_Model:
            print(f"[DEBUG - model] RMSNorm Input dtype: x = {x.dtype}")

        # If needed, cast weight to x's dtype and device for safe mixed precision
        weight = self.weight.to(dtype=x.dtype, device=x.device)
        
        # Compute root mean square
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        if Verbose_Model:
            print(f"[DEBUG - model] RMSNorm Output dtype: x = {x.dtype}, rms = {rms.dtype}")

        return weight * (x / (rms + self.eps))

## AdaLN 

class TimestepEmbedder(nn.Module):
    def __init__(self, d_model, t_emb_dim):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.freq_embedding_dim = t_emb_dim

        # Frequency embedding: sinusoidal
        self.lin1 = nn.Linear(t_emb_dim, t_emb_dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(t_emb_dim, d_model)
        nn.init.zeros_(self.lin2.weight)      # Zero-init the weights and bias of the linear2 layer
        nn.init.zeros_(self.lin2.bias)

    def forward(self, timestep):  # timestep: [B, 1, 1] or scalar
        device = timestep.device
        half_dim = self.freq_embedding_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        )
        # timestep: [B, 1, 1] → [B]
        t = timestep.view(-1).float()  # Ensures t is always [B]
        sinusoid = torch.outer(t, freqs)
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)  # shape: [B, t_emb_dim]
        # Feed through MLP
        emb = self.lin2(self.act(self.lin1(emb)))
        emb = emb.clamp(-10, 10) # Clamp the output to -10 to 10 to prevent the timestamp from being too erratically large
        return emb  # shape: [B, d_model]

class AdaLNModulator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 3),  # γ, β, α
        )
        # Custom initialization for last linear
        last_linear = self.ff[1]
        # γ and β bias to 0, α bias to 1
        nn.init.zeros_(last_linear.bias[:2*d_model])
        nn.init.constant_(last_linear.bias[2*d_model:], 1.0)
        # Weights remain as Kaiming (default)

    def forward(self, t_emb):
        return self.ff(t_emb).chunk(3, dim=-1)  # returns γ, β, α

## Other

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float, residual_scale: float = 1.0):
        super().__init__()
        self.norm = RMSNorm(features)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward(self, x, sublayer):
        # Pre-normalize input → pass through sublayer → dropout → residual add with scale
        return x + self.dropout(sublayer(self.norm(x))) * self.residual_scale
    
class FeedForward_SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2)  # Note: d_ff * 2!
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        if Verbose_Model:
            print(f"[DEBUG - model] FeedForward_SwiGLU Input dtype: x = {x.dtype}")
        x_proj = self.linear1(x)  # shape: (batch, seq, d_ff * 2)
        x_gated, x_linear = x_proj.chunk(2, dim=-1)  # Split into two halves
        x_act = F.silu(x_gated) * x_linear           # SwiGLU activation
        x_drop = self.dropout(x_act)
        if Verbose_Model:
            print(f"[DEBUG - model] FeedForward_SwiGLU Output dtype: x_drop = {x_drop.dtype}")
        return self.linear2(x_drop)

class FeedForward_AdaLN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.modulator = AdaLNModulator(d_model)
        self.linear1 = nn.Linear(d_model, d_ff * 2)  # SwiGLU: needs 2x d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        nn.init.zeros_(self.linear2.weight)   # Zero-init the weights and bias of the linear2 layer
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x, t_emb):

        if Verbose_Model:
            print(f"[DEBUG - FF_AdaLN] FeedForward_AdaLN Input dtype: x = {x.dtype}")

        # Step 1: Compute modulation params
        γ, β, α = self.modulator(t_emb)  # Each [B, D]
        γ = γ.clamp(-3.0, 3.0) # Clamp the output to -3.0 to 3.0 to give it bounds to maintain stability
        β = β.clamp(-5.0, 5.0) # ""  ""
        α = α.clamp(-3.0, 3.0) # ""  ""

        if torch.isnan(α).any() or torch.isinf(α).any():
            print(f"[NaN/Inf] Detected in α")
        if torch.isnan(γ).any() or torch.isinf(γ).any():
            print(f"[NaN/Inf] Detected in γ")
        if torch.isnan(β).any() or torch.isinf(β).any():
            print(f"[NaN/Inf] Detected in β")
        if Verbose_Stats_Modulator:
            print(f"[STATS - AdaLN Modulator] α mean: {α.mean().item():.5f}, std: {α.std().item():.5f}")
        if Verbose_Stats_Modulator:
            print(f"[STATS - AdaLN Modulator] γ mean: {γ.mean().item():.5f}, std: {γ.std().item():.5f}")
        if Verbose_Stats_Modulator:
            print(f"[STATS - AdaLN Modulator] β mean: {β.mean().item():.5f}, std: {β.std().item():.5f}")

        γ = γ.unsqueeze(1)  # [B, 1, D]
        β = β.unsqueeze(1)
        α = α.unsqueeze(1)

        α = α.clamp(-3.0, 3.0)
        γ = γ.clamp(-3.0, 3.0)
        β = β.clamp(-5.0, 5.0)
        if Verbose_Stats_ModulatorClamping:
            print(f"[DEBUG - Clamping-FF_AdaLN] FeedForward_AdaLN Modulator: α: mean={α.mean().item():.5f}, std={α.std().item():.5f}, max={α.abs().max().item():.2f} | γ: mean={γ.mean().item():.5f}, std={γ.std().item():.5f}, max={γ.abs().max().item():.2f} | β: mean={β.mean().item():.5f}, std={β.std().item():.5f}, max={β.abs().max().item():.2f}")

        # Step 2: Modulate input
        x_mod = (1 + γ) * x + β  # [B, T, D]

        # Step 3: SwiGLU MLP
        x_proj = self.linear1(x_mod)
        if Verbose_Stats_Modulator:
            print(f"[STATS - FF_AdaLN] FF x_proj std: {x_proj.std().item():.5f}")
        x_gated, x_linear = x_proj.chunk(2, dim=-1)
        x_act = F.silu(x_gated) * x_linear
        x_out = self.linear2(self.dropout(x_act))
        # x_out = torch.clamp(x_out, min=-5.0, max=5.0) # Clamp the output to -5.0 to 5.0 to prevent the exploding gradient problem
        x_out = torch.tanh(x_out) * 5.0 # Scratch that, this is even more aggressive to fight the exploding gradients

        if Verbose_Model:
            print(f"[DEBUG - model] FeedForward_AdaLN Output dtype: x = {x.dtype}, α = {α.dtype}, x_out = {x_out.dtype}")

        # Step 4: Residual with AdaLN α gate
        return x + α * x_out
        
# RoPE-related functions and storage

def generate_sin_cos(seq_len, dim, device):
    half_dim = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    positions = torch.arange(seq_len, device=device).float()
    sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    sin = torch.cat([sin, sin], dim=-1)  # expand to full dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)
    if Verbose_Model:
        if sin.dtype != torch.float32 or cos.dtype != torch.float32:
            print(f"[WARN] RoPE sin/cos dtype not float32! sin: {sin.dtype}, cos: {cos.dtype}")
        else:
            print(f"[DEBUG - model] RoPE sin/cos dtype: {sin.dtype}, {cos.dtype}")
    return sin.unsqueeze(0), cos.unsqueeze(0)  # [1, seq_len, dim]

def rotate(x):
    x1 = x[..., ::2]  # even dims
    x2 = x[..., 1::2]  # odd dims
    return torch.stack([-x2, x1], dim=-1).reshape_as(x)

def apply_rope_to(q, k, sin, cos):
    # Ensure sin/cos are same dtype and shape as q/k
    sin = sin[:, :q.shape[1], :].to(dtype=q.dtype, device=q.device)
    cos = cos[:, :q.shape[1], :].to(dtype=q.dtype, device=q.device)
    q_rot = q * cos + rotate(q) * sin
    k_rot = k * cos + rotate(k) * sin
    return q_rot, k_rot

def build_causal_mask(size, device):
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

## Attention Blocks

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.bool() # Guarantee mask is bool (needed for ~mask)
            attention_scores.masked_fill_(~mask, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class GeneralAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.core = MultiHeadAttentionBlock(d_model, n_heads, dropout)

    def forward(self, q, k, v, mask=None):
        if self.use_rope:
            sin, cos = generate_sin_cos(seq_len=q.size(1), dim=q.size(-1), device=q.device)
            q, k = apply_rope_to(q, k, sin, cos)
            if Verbose_Model:
                print(f"[DEBUG - Attention-RoPE] RoPE sin/cos dtype in attention block: {sin.dtype}, {cos.dtype}")
        return self.core(q, k, v, mask)

class ContextualizerSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = GeneralAttention(d_model, n_heads, dropout, use_rope=True)

    def forward(self, x):
        if Verbose_Contextualizer:
            print(f"[DEBUG - Attention-CxSA] ContextualSelfAttention Input dtype: x = {x.dtype}")
        causal_mask = build_causal_mask(x.shape[1], x.device)
        if Verbose_Contextualizer:
            print(f"[DEBUG - Attention-CxSA] causal_mask dtype: {causal_mask.dtype}")
            print(f"[DEBUG - Attention-CxSA] padding_mask dtype: {self.padding_mask.dtype}")
        full_mask = causal_mask.bool() & self.padding_mask.unsqueeze(1).unsqueeze(2).bool() # Combine the causal and padding masks to ensure the model doesn't attend to either future or padding tokens... and make sure the masks are bool first
        if Verbose_Contextualizer:
            print(f"[DEBUG - Attention-CxSA] ContextualSelfAttention Mask dtype: x = {x.dtype}, mask = {full_mask.dtype}")
        return self.attn(x, x, x, full_mask)

class DenoiserSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = GeneralAttention(d_model, n_heads, dropout, use_rope=False)
        self.modulator = AdaLNModulator(d_model)

    def forward(self, x, t_emb):

        if Verbose_Denoiser:
            print(f"[DEBUG - Attention-DnSA] DenoiserSelfAttention Input dtype: x = {x.dtype}")

        # 1. Modulate input with AdaLN based on timestep
        γ, β, α = self.modulator(t_emb) # [batch, d_model] each
        γ = γ.clamp(-3.0, 3.0) # Clamp the output to -3.0 to 3.0 to give it bounds to maintain stability
        β = β.clamp(-5.0, 5.0) # ""  ""
        α = α.clamp(-3.0, 3.0) # ""  ""
        if Verbose_Stats_ModulatorClamping:
            print(f"[DEBUG - Clamping-DnSA] DenoiserSelfAttention Modulator: α: mean={α.mean().item():.5f}, std={α.std().item():.5f}, max={α.abs().max().item():.2f} | γ: mean={γ.mean().item():.5f}, std={γ.std().item():.5f}, max={γ.abs().max().item():.2f} | β: mean={β.mean().item():.5f}, std={β.std().item():.5f}, max={β.abs().max().item():.2f}")
        γ = γ.unsqueeze(1) # -> [batch, 1, d_model]
        β = β.unsqueeze(1) # -> [batch, 1, d_model]
        α = α.unsqueeze(1) # -> [batch, 1, d_model]

        # 2. Apply AdaLN modulation
        x_mod = (1 + γ) * x + β

        # 3. Create causal mask [1, 1, seq_len, seq_len]
        seq_len = x.shape[1]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(1)
        if Verbose_Denoiser:
            print(f"[DEBUG - Attention-DnSA] causal_mask dtype: {causal_mask.dtype}")
            print(f"[DEBUG - Attention-DnSA] padding_mask dtype: {self.padding_mask.dtype}")
        full_mask = causal_mask * self.padding_mask.unsqueeze(1).unsqueeze(2) # Combine the causal and padding masks to ensure the model doesn't attend to either future or padding tokens

        # 4. Run attention (Q = K = V = x_mod)
        y = self.attn(x_mod, x_mod, x_mod, full_mask)

        if Verbose_Denoiser:
            print(f"[DEBUG - Attention-DSA] DenoiserSelfAttention Output dtype: x = {x.dtype}, α = {α.dtype}, y = {y.dtype}")

        # 5. Apply residual connection and scaling
        return x + α * y
    
class DenoiserCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = GeneralAttention(d_model, n_heads, dropout, use_rope=False)
        self.modulator = AdaLNModulator(d_model)

    def forward(self, x, context, t_emb, *, dropout_denoiser=0.0, training=False):

        if Verbose_Denoiser:
            print(f"[DEBUG - Attention-DnCA] DenoiserCrossAttention Input dtype: x = {x.dtype}")

        # 1. Modulate input with AdaLN based on timestep
        γ, β, α = self.modulator(t_emb) # [batch, d_model] each
        γ = γ.clamp(-3.0, 3.0) # Clamp the output to -3.0 to 3.0 to give it bounds to maintain stability
        β = β.clamp(-5.0, 5.0) # ""  ""
        α = α.clamp(-3.0, 3.0) # ""  ""
        if Verbose_Stats_ModulatorClamping:
            print(f"[DEBUG - Clamping-DnCA] DenoiserCrossAttention Modulator: α: mean={α.mean().item():.5f}, std={α.std().item():.5f}, max={α.abs().max().item():.2f} | γ: mean={γ.mean().item():.5f}, std={γ.std().item():.5f}, max={γ.abs().max().item():.2f} | β: mean={β.mean().item():.5f}, std={β.std().item():.5f}, max={β.abs().max().item():.2f}")
        γ = γ.unsqueeze(1) # -> [batch, 1, d_model]
        β = β.unsqueeze(1) # -> [batch, 1, d_model]
        α = α.unsqueeze(1) # -> [batch, 1, d_model]

        # 2. Apply AdaLN modulation
        x_mod = (1 + γ) * x + β

        # 3. Prepend zero-vector to context, which provides the position 0 something to attend to
        zero = torch.zeros((context.size(0), 1, context.size(2)), device=context.device, dtype=context.dtype)
        context = torch.cat([zero, context], dim=1)

        # 3.1 Prepend zero-vector to padding mask
        padding_mask = self.padding_mask
        padding_mask = torch.cat([
            torch.ones((padding_mask.size(0), 1), dtype=padding_mask.dtype, device=padding_mask.device), 
            padding_mask
        ], dim=1)

        if Verbose_Denoiser:
            print(f"[DEBUG - Attention-DnCA] padded_mask shape: {padding_mask.shape}, dtype: {padding_mask.dtype}")
            print(f"[DEBUG - Attention-DnCA] padded_mask[0, :10]: {padding_mask[0, :10]}")  # print first 10 for one sample

        # 4. Build causal mask for the context sequence
        seq_len_q = x_mod.size(1)
        seq_len_k = context.size(1)
        causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_k), device=x.device)).unsqueeze(0).unsqueeze(1)
        if Verbose_Denoiser:
            print(f"[DEBUG - Attention-DnCA] causal_mask dtype: {causal_mask.dtype}")
            print(f"[DEBUG - Attention-DnCA] padding_mask dtype: {padding_mask.dtype}")
        causal_and_padding_mask = causal_mask.bool() & padding_mask.unsqueeze(1).unsqueeze(2).bool() # Combine the causal and padding masks to ensure the model doesn't attend to either future or padding tokens... and make sure the masks are bool first

        # 5. Apply Row-Level CFG Dropout
        full_mask = causal_and_padding_mask  # default fallback mask
        if training and dropout_denoiser > 0.0:
            keep_mask = (torch.rand(context.size(0), context.size(1), device=context.device) > dropout_denoiser)
            keep_mask[:, 0] = True
            context = context * keep_mask.unsqueeze(-1)
            full_mask = causal_and_padding_mask & keep_mask.unsqueeze(1).unsqueeze(2) # Apply the keep_mask to the full_mask to add the CFG mask

        # 6. Run attention
        y = self.attn(x_mod, context, context, full_mask)

        if Verbose_Denoiser:
            print(f"[DEBUG - Attention-DCA] DenoiserCrossAttention Output dtype: x = {x.dtype}, α = {α.dtype}, y = {y.dtype}")

        # 7. Apply residual connection and scaling
        return x + α * y


## ------------------------------------------------------------
## Main Layers
## ------------------------------------------------------------

## PreNets and PostNets

class PreNetC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return x

class PostNetC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

class PreNetD(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return x

class PostNetD(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

## Contextualizer Tower

class ContextualizerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, residual_scale: float):
        super().__init__()
        self.self_attention = ContextualizerSelfAttention(d_model, n_heads, dropout)
        self.mlp = FeedForward_SwiGLU(d_model, d_ff, dropout)

        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout, residual_scale),
            ResidualConnection(d_model, dropout, residual_scale)
        ])

    def forward(self, x):
        x = self.residual_connections[0](x, self.self_attention)
        x = self.residual_connections[1](x, self.mlp)
        return x

class ContextualizerTower(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        scale = 1.0 / math.sqrt(num_layers)
        self.norm = RMSNorm(d_model)  # Final norm layer (post-residual stack)
        self.layers = nn.ModuleList([
            ContextualizerLayer(d_model, n_heads, d_ff, dropout, scale)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if Verbose_Model:
                print(f"[DEBUG - model] Before ContextualizerLayer {i}: dtype = {x.dtype}")
            x = layer(x)
            if Verbose_Model:
                print(f"[DEBUG - model] After ContextualizerLayer {i}: dtype = {x.dtype}")
        x = self.norm(x)
        if Verbose_Model:
            print(f"[DEBUG - model] After ContextualizerTower norm (before dtype clamp): dtype = {x.dtype}")
        x = x.to(torch.bfloat16) # Clamp back to bf16 bacause the fp32 RMS value causes it to be promoted to fp32
        if Verbose_Model:
            print(f"[DEBUG - model] After ContextualizerTower norm: dtype = {x.dtype}")
        return x

## Latent Bridge

class LatentBridge(nn.Module): # ToDo: Add in future functionality such as the gate for MoE or activation steering, but for now it's just a simple pass-through layer
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

## Denoiser Tower

class DenoiserLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float, residual_scale: float = 1.0):
        super().__init__()
        self.self_attention = DenoiserSelfAttention(d_model, n_heads, dropout)
        self.cross_attention = DenoiserCrossAttention(d_model, n_heads, dropout)
        self.mlp = FeedForward_AdaLN(d_model, d_ff, dropout)

        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout, residual_scale),
            ResidualConnection(d_model, dropout, residual_scale),
            ResidualConnection(d_model, dropout, residual_scale)
        ])

    def forward(self, x, context, timestep, *, dropout_denoiser=0.0, training=False):
        # Mask should be externally assigned before call. Each sublayer already handles AdaLN inside
        x = self.residual_connections[0](x, lambda x_: self.self_attention(x_, timestep))
        x = self.residual_connections[1](x, lambda x_: self.cross_attention(x_, context, timestep, dropout_denoiser=dropout_denoiser, training=training))
        x = self.residual_connections[2](x, lambda x_: self.mlp(x_, timestep))
        return x

class DenoiserTower(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.final_norm = RMSNorm(d_model) # Optional — depends on paper interpretation
        residual_scale = 1.0 / math.sqrt(num_layers)

        self.layers = nn.ModuleList([
            DenoiserLayer(d_model, d_ff, n_heads, dropout, residual_scale)
            for _ in range(num_layers)
        ])

    def forward(self, x, context, timestep, *, dropout_denoiser=0.0, training=False):
        with autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
            for i, layer in enumerate(self.layers):
                x = layer(x, context, timestep, dropout_denoiser=dropout_denoiser, training=training)
                if Verbose_Model:
                    print(f"[DEBUG - model] After DenoiserLayer {i}: dtype = {x.dtype}, mean = {x.mean().item():.5f}, std = {x.std().item():.5f}")
                if Verbose_Stats_ModulatorClamping and i == 2:
                    # This is Layer 2, log α/γ/β from the self-attn modulator
                    t_emb = timestep  # [B, d_model], matches your forward call
                    γ, β, α = layer.self_attention.modulator(t_emb)
                    print(f"[DEBUG - DenoiserTower] Denoiser Post-Layer α/β/γ: α: mean={α.mean().item():.5f}, std={α.std().item():.5f}, max={α.abs().max().item():.2f} | γ: mean={γ.mean().item():.5f}, std={γ.std().item():.5f}, max={γ.abs().max().item():.2f} | β: mean={β.mean().item():.5f}, std={β.std().item():.5f}, max={β.abs().max().item():.2f}")

            x = self.final_norm(x)

            if Verbose_Model:
                print(f"[DEBUG - model] After DenoiserTower norm (before dtype clamp): dtype = {x.dtype}")
            x = x.to(torch.bfloat16)
            if Verbose_Model:
                print(f"[DEBUG - model] After DenoiserTower norm: dtype = {x.dtype}")

        assert x.dtype == torch.bfloat16, f"Dtype drifted at DenoiserTower output: {x.dtype}"

        return x

## ------------------------------------------------------------
## Loss Functions
## ------------------------------------------------------------

def l2_euclidean_loss_with_mask(
        predicted: torch.Tensor,       # [B, T, D] or [B, D]
        target: torch.Tensor,          # [B, T, D] or [B, D]
        attention_mask: torch.Tensor   # [B, T] or [B, 1]
    ) -> torch.Tensor:
    """
    Computes mean L2 (Euclidean) distance between predicted and target at every non-masked token position.
    """
    if Verbose_Loss:
        print(f"[DEBUG - Loss] l2_euclidean_loss_with_mask: predicted.shape = {predicted.shape}, target.shape = {target.shape}, attention_mask.shape = {attention_mask.shape}")

    # Ensure all tensors are 3D (add time dim if needed)
    if predicted.dim() == 2:
        predicted = predicted.unsqueeze(1)
    if target.dim() == 2:
        target = target.unsqueeze(1)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(1)

    # Make shapes match (truncate to min sequence length)
    T = min(predicted.shape[1], target.shape[1], attention_mask.shape[1])
    predicted = predicted[:, :T, :]
    target = target[:, :T, :]
    attention_mask = attention_mask[:, :T]

    # Compute L2 norm at every position: [B, T]
    l2 = torch.norm(predicted - target, dim=-1)  # [B, T]
    # Zero out padded positions
    l2 = l2 * attention_mask  # [B, T]
    # Compute mean over all valid (non-masked) tokens
    total_valid = attention_mask.sum()
    # Avoid div by zero
    if total_valid == 0:
        return l2.mean()  # fallback, should never trigger if mask is good
    return l2.sum() / total_valid

## ------------------------------------------------------------
## LexaLCM Model's Main Architecture
## ------------------------------------------------------------

class LexaLCM(PreTrainedModel):
    config_class = LexaLCMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Precompute cosine schedule for denoising loop
        max_denoising_steps = max(
            config.denoiser_iterations_pretrain,
            config.denoiser_iterations_inference
        )
        self.register_buffer(
            "alpha_bar_schedule",
            self._compute_cosine_schedule(max_denoising_steps)
        )

        # Create TimestepEmbedder for AdaLN
        self.TimestepEmbedder = TimestepEmbedder(t_emb_dim=config.AdaLN_Timestep_Embed_Dim, d_model=config.d_model)

        # Architecture

        self.PreNet_C_Up = PreNetC(config.input_dim, config.d_model)

        self.ContextualizerTower = ContextualizerTower(
            num_layers=config.num_context_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout_context
        )

        self.PostNet_C_Down = PostNetC(config.d_model, config.d_latent)

        self.LatentBridge = LatentBridge()

        self.PreNet_D_Up = PreNetD(config.d_latent, config.d_model)

        self.DenoiserTower = DenoiserTower(
            num_layers=config.num_denoiser_layers,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            dropout=config.dropout_denoiser
        )

        self.PostNet_D_Down = PostNetD(config.d_model, config.input_dim)

    def _compute_cosine_schedule(self, num_steps: int, s: float = 0.008):
        """
        Precompute the cosine noise schedule: alpha_bar[t] = cos^2( (t/T + s) / (1+s) * π/2 )
        """
        steps = torch.arange(num_steps + 1, dtype=torch.float64)  # T+1 values
        f = (steps / num_steps + s) / (1 + s)
        alpha_bar = torch.cos(f * math.pi / 2) ** 2

        if Verbose_Model:
            print(f"[DEBUG - model] alpha_bar[0] = {alpha_bar[0].item():.6f}")
            print(f"[DEBUG - model] alpha_bar[-1] = {alpha_bar[-1].item():.6f}")

        return alpha_bar.to(torch.float32)  # [T+1]

    def run_denoising_loop(
        self,
        context,
        *,
        training=False,
        dropout_denoiser=0.0,
        cfg_scale: float = 0.0, # Classifier-Free Guidance Scale
        uncond_context: torch.Tensor = None, # Unconditional context for CFG
    ):
        """
        Perform iterative denoising using the DenoiserTower.
        
        Args:
            context: [B, T, D] fixed contextualizer output
            training: bool, whether we're training (affects # iterations and dropout)
            dropout_denoiser: float, optional dropout rate used during classifier-free guidance training

        Returns:
            denoised_latents: [B, T, D]
        """

        x_0 = context.detach()  # Target latents (constant across timesteps)

        num_steps = (
            self.config.denoiser_iterations_pretrain if training 
            else self.config.denoiser_iterations_inference
        )

        for t in range(num_steps):
            # 1. Get alpha_bar_t
            alpha_bar = self.alpha_bar_schedule[t].to(x_0.dtype)
            if Verbose_Model:
                print(f"[DEBUG - model] ᾱ[{t}] = {alpha_bar.item():.6f}")

            # 2. Sample new noise each step
            epsilon = torch.randn_like(x_0)

            # 3. Forward diffuse both mix x_0 and epsilon
            x_t_cond = (alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * epsilon)

            # 4. Embed timestep and denoise
            timestep = torch.full(
                (x_0.shape[0], 1, 1),
                fill_value=t,
                dtype=torch.float32,
                device=x_0.device
            )
            t_emb = self.TimestepEmbedder(timestep).to(x_0.dtype)

            if cfg_scale > 0.0 and not training: # If CFG is enabled and we're not training, we need to denoise both the conditional and unconditional contexts
                x_t_uncond = (alpha_bar.sqrt() * uncond_context + (1 - alpha_bar).sqrt() * epsilon)

                x_cond = self.DenoiserTower(
                    x_t_cond, 
                    x_0, 
                    t_emb,
                    dropout_denoiser=0.0, training=False
                )
                x_uncond = self.DenoiserTower(
                    x_t_uncond, uncond_context, t_emb,
                    dropout_denoiser=0.0, training=False
                )

                # CFG interpolation
                x = x_uncond + cfg_scale * (x_cond - x_uncond)

            else:
                # Training or no CFG
                x = self.DenoiserTower(
                    x_t_cond, 
                    x_0,
                    t_emb,
                    dropout_denoiser=dropout_denoiser,
                    training=training
                )

            if Verbose_Model:
                print(f"[DEBUG - model] Denoising Loop #{t}")
                if t % 10 == 0:
                    print(f"[DEBUG - model] Denoising Loop step {t} ᾱ = {alpha_bar.item():.6f}")


        return x

    def forward(self, embeddings, labels=None, attention_mask=None, **kwargs):
        if Verbose_Loss:
            print(f"[DEBUG - Loss] embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")

        # Convert attention_mask [B, T] into boolean mask where True = keep, False = pad
        padding_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(embeddings[:, :, 0]).bool()

        # Assign padding mask to attention modules
        for layer in self.ContextualizerTower.layers:
            layer.self_attention.padding_mask = padding_mask

        for layer in self.DenoiserTower.layers:
            layer.self_attention.padding_mask = padding_mask
            layer.cross_attention.padding_mask = padding_mask

        inspection_decoder = getattr(self, "inspection_decoder", None)
        if inspection_decoder is not None:
            inspect_util(embeddings, inspection_decoder, num_batches=1, num_seqs=8) # ToDo: make this configurable
        
        # PreNet - Contextualizer Tower

        x = self.PreNet_C_Up(embeddings)
        if Verbose_Model:
            print(f"[DEBUG - model] after PreNet_C_Up: shape={x.shape}, dtype={x.dtype}")

        # Contextualizer Tower

        x = self.ContextualizerTower(x)
        if Verbose_Model:
            print(f"[DEBUG - model] after ContextualizerTower: shape={x.shape}, dtype={x.dtype}")

        # PostNet - Contextualizer Tower

        x = self.PostNet_C_Down(x)
        if Verbose_Model:
            print(f"[DEBUG - model] after PostNet_C_Down: shape={x.shape}, dtype={x.dtype}")

        # LatentBridge

        x = self.LatentBridge(x)
        if Verbose_Model:
            print(f"[DEBUG - model] after LatentBridge: shape={x.shape}, dtype={x.dtype}")

        # PreNet - DenoiserTower

        x = self.PreNet_D_Up(x)
        if Verbose_Model:
            print(f"[DEBUG - model] after PreNet_D_Up: shape={x.shape}, dtype={x.dtype}")

        # Denoising Loop

        # Project to denoiser input space (i.e., noise dimension)
        latent_context = x  # [B, T, D]

        # Create unconditional context for CFG if CFG is enabled and we're not training
        uncond_context = None 
        if not self.training and self.config.cfg_scale > 0.0:
            # Create unconditional context by copying the first token of the conditional context
            uncond_context = torch.zeros_like(latent_context)
            uncond_context[:, 0, :] = latent_context[:, 0, :]

        # Denoise from noise → latents using contextual embedding
        x = self.run_denoising_loop(
            context=latent_context,
            training=self.training,
            dropout_denoiser=self.config.dropout_denoiser,
            cfg_scale=self.config.cfg_scale,
            uncond_context=uncond_context,
        )

        if Verbose_Model:
            print(f"[DEBUG - model] after Denoising Loop: shape={x.shape}, dtype={x.dtype}")

        # PostNet - DenoiserTower

        with autocast(device_type="cuda", enabled=False):
            x = x.to(torch.float32)
            if Verbose_Model:
                print(f"[DEBUG - model] after to(float32) PostNet_D_Down: shape={x.shape}, dtype={x.dtype}")
            x = self.PostNet_D_Down(x)
            if Verbose_Model:
                print(f"[DEBUG - model] after PostNet_D_Down: shape={x.shape}, dtype={x.dtype}")

        if inspection_decoder is not None:
            inspect_util(x, inspection_decoder, num_batches=1, num_seqs=8) # ToDo: make this configurable

        if Verbose_Model:
            print(f"[DEBUG - model] final output: shape={x.shape}, dtype={x.dtype}")

        if labels is not None:
            # labels: [B, T, D], x: [B, T, D], attention_mask: [B, T]
            if Verbose_Model:   
                print(f"[DEBUG - model] labels is not None, returning loss and logits - shape={x.shape}, dtype={x.dtype}")
            return {
                "loss": l2_euclidean_loss_with_mask(x, labels, attention_mask),
                "logits": x,  # Keep all timesteps for analysis
            }

        # if labels is not None:
        #     labels = labels.unsqueeze(1)  # [B, 1, D]

        #     # Shape fix: model returns [B, D] but loss expects [B, T, D]
        #     x = x.unsqueeze(1)           # [B, 1, D]
        #     labels = labels.unsqueeze(1) # [B, 1, D]
        #     attention_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)  # [B, 1]
        #     if Verbose_Model:   
        #         print(f"[DEBUG - model] labels is not None, returning loss and logits - shape={x.shape}, dtype={x.dtype}")

        #     return {
        #         "loss": l2_euclidean_loss_with_mask(x, labels, attention_mask),
        #         "logits": x.squeeze(1),  # Optional: remove fake T dimension for outputs
        #     }
        # else:
        #     print(f"[DEBUG - model] labels is None, returning x[:, -1:, :] - shape={x[:, -1:, :].shape}, dtype={x[:, -1:, :].dtype}")
        #     if Verbose_Loss:
        #         print(f"[DEBUG - Loss] Loss is {l2_euclidean_loss_with_mask(x, embeddings, torch.ones_like(embeddings[:, :, 0]).bool())}")
        #     return x[:, -1:, :]
        else:
            print(f"[DEBUG - model] labels is None, likely being used for inference. Returning predictied embeddings - shape={x.shape}, dtype={x.dtype}")
            return x

MODEL_MAPPING.register(LexaLCMConfig, LexaLCM)