# gdntpu/delta_rule.py
# Pure-PyTorch / XLA-compatible Gated Delta Rule recurrence.
#
# Implements two modes that mirror FLA exactly:
#   1. pure_chunk_gated_delta_rule    — training path (chunk-parallel)
#   2. pure_recurrent_gated_delta_rule — inference path (step-by-step)
#
# Mathematical reference (paper Eq. 10):
#   S_t = S_{t-1} * α_t * (I − β_t k_t k_t^T) + β_t v_t k_t^T
# where α_t = exp(g_t),  g_t = −exp(A_log) * softplus(g_raw + dt_bias).
#
# Source of truth for numerics: support_files/gated_delta_rule/naive.py

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helper: compute log-space gate  g = −exp(A_log) * softplus(g_raw + dt_bias)
# ---------------------------------------------------------------------------
def _compute_gate(
    g_raw: torch.Tensor,          # [B, T, HV]
    A_log: torch.Tensor,          # [HV]
    dt_bias: torch.Tensor,        # [HV]
) -> torch.Tensor:
    """Fused GDN gate (gate.py naive_gdn_gate, line 44).

    Returns log-space decay values (negative, so exp() gives values in (0,1)).
    Shape: same as g_raw = [B, T, HV].
    """
    g = g_raw.float()                              # up-cast for stability
    g = g + dt_bias.float()                        # add dt_bias
    A_exp = A_log.float().exp()                    # exp(A_log)
    return (-A_exp * F.softplus(g)).to(g_raw.dtype)


# ---------------------------------------------------------------------------
# Recurrent (step-by-step) — used at inference when q_len ≤ 64
# ---------------------------------------------------------------------------
def pure_recurrent_gated_delta_rule(
    q: torch.Tensor,               # [B, T, H, K]
    k: torch.Tensor,               # [B, T, H, K]
    v: torch.Tensor,               # [B, T, HV, V]
    g: torch.Tensor,               # [B, T, HV]   — already in log space
    beta: torch.Tensor,            # [B, T, HV]
    scale: float,
    initial_state: torch.Tensor | None = None,  # [B, HV, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Step-by-step recurrent Gated Delta Rule.

    Numerically matches FLA's ``naive_recurrent_gated_delta_rule`` (naive.py
    lines 13-64).  The inner loop is unrolled by XLA when T is static.

    Note: ``q`` and ``k`` must already be L2-normalised before calling this
    function.  Scale is applied to q inside this function.

    Returns:
        o     : [B, T, HV, V]
        state : [B, HV, K, V]  (or None if output_final_state=False)
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    # Transpose to [B, H/HV, T, dim] for the loop — matches naive.py layout.
    # naive.py line 40: q,k,v,beta,g = ... .transpose(1,2).contiguous().to(float32)
    q   = q.permute(0, 2, 1, 3).float()    # [B, H,  T, K]
    k   = k.permute(0, 2, 1, 3).float()    # [B, H,  T, K]
    v   = v.permute(0, 2, 1, 3).float()    # [B, HV, T, V]
    g   = g.permute(0, 2, 1).float()       # [B, HV, T]
    beta = beta.permute(0, 2, 1).float()   # [B, HV, T]

    q = q * scale  # naive.py line 48: q = q * scale

    # Allocate output and state.
    o = torch.zeros(B, HV, T, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        h = initial_state.float()          # [B, HV, K, V]
    else:
        h = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)

    # GVA: q/k have H heads, v/beta/g have HV heads. HV >= H, HV % H == 0.
    gva_ratio = HV // H

    for t in range(T):
        # Decay the state: h *= exp(g_t)   (naive.py line 54)
        g_t = g[:, :, t]                          # [B, HV]
        h = h * g_t.exp().unsqueeze(-1).unsqueeze(-1)   # [B, HV, K, V]

        # Repeat q/k for GVA groups if needed.
        q_t = q[:, :, t]                          # [B, H, K]
        k_t = k[:, :, t]                          # [B, H, K]
        if gva_ratio > 1:
            q_t = q_t.repeat_interleave(gva_ratio, dim=1)  # [B, HV, K]
            k_t = k_t.repeat_interleave(gva_ratio, dim=1)  # [B, HV, K]

        v_t  = v[:, :, t].clone()                 # [B, HV, V]
        b_t  = beta[:, :, t]                      # [B, HV]

        # Delta erase: v_new = v − h @ k   (naive.py line 56)
        # h @ k_t: [B, HV, K, V] * k_t[B, HV, K, 1] → sum over K → [B, HV, V]
        v_t  = v_t - (h * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, HV, V]

        # Write-strength: v_new *= beta    (naive.py line 57)
        v_t  = v_t * b_t.unsqueeze(-1)            # [B, HV, V]

        # State update: h += k^T ⊗ v_new  (naive.py line 58)
        h = h + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # [B, HV, K, V]

        # Readout: o_t = q @ h            (naive.py line 59)
        o[:, :, t] = (q_t.unsqueeze(-2) @ h).squeeze(-2)  # [B, HV, V]

    # [B, HV, T, V] → [B, T, HV, V]
    o = o.permute(0, 2, 1, 3).to(v.dtype)

    final_state = h if output_final_state else None
    return o, final_state


# ---------------------------------------------------------------------------
# Chunk (parallel training) — used during training
# ---------------------------------------------------------------------------
def pure_chunk_gated_delta_rule(
    q: torch.Tensor,              # [B, T, H, K]
    k: torch.Tensor,              # [B, T, H, K]
    v: torch.Tensor,              # [B, T, HV, V]
    g: torch.Tensor,              # [B, T, HV]  — log-space decay
    beta: torch.Tensor,           # [B, T, HV]
    scale: float,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,   # [B, HV, K, V]
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunk-parallel Gated Delta Rule for hardware-efficient training.

    Implements the chunk algorithm from naive.py ``naive_chunk_gated_delta_rule``
    (lines 67-161).  The chunk decomposition replaces the O(T) sequential loop
    with O(T/C) outer chunk-steps, each using dense matmuls over chunk size C.

    All intermediates are fp32; output is cast back to input dtype.

    Returns:
        o     : [B, T, HV, V]
        state : [B, HV, K, V] or None
    """
    BT = chunk_size
    B, T, H, K = q.shape
    HV = v.shape[2]
    V  = v.shape[3]

    # ---- Pad to multiple of BT ------------------------------------------
    pad_len = (BT - T % BT) % BT
    if pad_len > 0:
        # Pad time dimension for each tensor.
        q    = F.pad(q,    (0, 0, 0, 0, 0, pad_len))   # [B, T+P, H,  K]
        k    = F.pad(k,    (0, 0, 0, 0, 0, pad_len))   # [B, T+P, H,  K]
        v    = F.pad(v,    (0, 0, 0, 0, 0, pad_len))   # [B, T+P, HV, V]
        beta = F.pad(beta, (0, 0, 0, pad_len))          # [B, T+P, HV]
        g    = F.pad(g,    (0, 0, 0, pad_len))          # [B, T+P, HV]

    T_pad = T + pad_len

    # Cast all to float32 (naive.py lines 100-111).
    q    = q.float()      # [B, T_pad, H,  K]
    k    = k.float()      # [B, T_pad, H,  K]
    v    = v.float()      # [B, T_pad, HV, V]
    beta = beta.float()   # [B, T_pad, HV]
    g    = g.float()      # [B, T_pad, HV]  (already log-space)

    # ---- Transpose to [B, H/HV, T_pad, dim] ----------------------------
    q    = q.permute(0, 2, 1, 3)     # [B, H,  T_pad, K]
    k    = k.permute(0, 2, 1, 3)     # [B, H,  T_pad, K]
    v    = v.permute(0, 2, 1, 3)     # [B, HV, T_pad, V]
    beta = beta.permute(0, 2, 1)     # [B, HV, T_pad]
    g    = g.permute(0, 2, 1)        # [B, HV, T_pad]

    q = q * scale   # naive.py line 116

    # GVA: expand q/k to match HV heads.
    gva_ratio = HV // H
    if gva_ratio > 1:
        q = q.repeat_interleave(gva_ratio, dim=1)  # [B, HV, T_pad, K]
        k = k.repeat_interleave(gva_ratio, dim=1)  # [B, HV, T_pad, K]

    # v_beta and k_beta (naive.py lines 117-118)
    v_beta = v * beta.unsqueeze(-1)      # [B, HV, T_pad, V]
    k_beta = k * beta.unsqueeze(-1)      # [B, HV, T_pad, K]

    NC = T_pad // BT  # number of chunks

    # Reshape into chunks: [B, HV, NC, BT, dim]
    def _chunk(x, last_dim):
        return x.view(B, x.shape[1], NC, BT, last_dim)

    q_c      = _chunk(q,      K)   # [B, HV, NC, BT, K]
    k_c      = _chunk(k,      K)
    v_c      = _chunk(v,      V)
    beta_c   = _chunk(beta.unsqueeze(-1), 1).squeeze(-1)  # [B, HV, NC, BT]
    g_c      = _chunk(g.unsqueeze(-1), 1).squeeze(-1)     # [B, HV, NC, BT]
    v_beta_c = _chunk(v_beta, V)
    k_beta_c = _chunk(k_beta, K)

    # Cumulative sum of log-decay within each chunk (naive.py line 127).
    # decay_c[..., r] = sum_{i=0}^{r} g_c[..., i]
    decay_c = g_c.cumsum(dim=-1)           # [B, HV, NC, BT]
    # Exponential decay matrix within chunk: L_mask[i,j] = exp(decay[i]-decay[j])
    # Only lower-triangular (causal). naive.py line 129.
    decay_diff = decay_c.unsqueeze(-1) - decay_c.unsqueeze(-2)  # [B,HV,NC,BT,BT]
    L_mask = torch.tril(decay_diff.exp())                        # [B,HV,NC,BT,BT]

    # ---- Intra-chunk: WY representation ---------------------------------
    # naive.py lines 130-136: solve lower-triangular system for attn matrix.
    # attn = (I + lower_tril(beta * K @ K^T * L_mask))^{-1} * diag(beta)
    # We follow the naive sequential solve: for each row i, subtract the sum
    # of previous rows scaled by attn[i, :i] @ attn[:i, :i].
    mask_upper = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0
    )  # strictly upper-triangular+diagonal mask for zeroing
    mask_diag_excl = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1
    )  # strictly above diagonal

    # Raw interaction matrix: beta * K @ K^T * L_mask, strictly lower-tri.
    # shape: [B, HV, NC, BT, BT]
    KKt = k_beta_c @ k_c.transpose(-1, -2)  # [B, HV, NC, BT, BT]
    A = -(KKt * L_mask).masked_fill(mask_upper, 0)  # naive.py line 130

    # Sequential forward substitution (naive.py lines 131-132).
    # For row i: A[i,:i] += A[i,:i] @ A[:i,:i]  (accumulate over prior rows)
    # Note: .clone() removed — explicit indexing avoids in-place grad issues,
    # and the matmul result is a new tensor, so no aliasing conflict.
    for i in range(1, BT):
        # A[..., i:i+1, :i] @ A[..., :i, :i] → [B, HV, NC, 1, i]
        correction = (A[..., i:i+1, :i] @ A[..., :i, :i]).squeeze(-2)
        A = A.clone()  # single clone per iteration for autograd safety
        A[..., i, :i] = A[..., i, :i] + correction

    # Add identity to get (I + A)^{-1} representation (naive.py line 133).
    eye = torch.eye(BT, dtype=torch.float32, device=q.device)
    A = A + eye  # [B, HV, NC, BT, BT]

    # Effective new values and key-decay (naive.py lines 135-137).
    # u = A @ (beta * v)   →  updated values
    # w = A @ (beta * k * exp(decay))  → decayed key weights for inter-chunk
    decay_exp_c = decay_c.exp().unsqueeze(-1)   # [B, HV, NC, BT, 1]
    u = A @ v_beta_c                             # [B, HV, NC, BT, V]
    w = A @ (k_beta_c * decay_exp_c)            # [B, HV, NC, BT, K]

    # ---- Inter-chunk: state propagation ---------------------------------
    # Allocate state and output.
    S = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        S = initial_state.float()  # [B, HV, K, V]

    o_c = torch.zeros(B, HV, NC, BT, V, dtype=torch.float32, device=q.device)

    mask_upper_o = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1
    )

    for i in range(NC):
        q_i   = q_c[:, :, i]       # [B, HV, BT, K]
        k_i   = k_c[:, :, i]       # [B, HV, BT, K]
        u_i   = u[:, :, i]         # [B, HV, BT, V]
        w_i   = w[:, :, i]         # [B, HV, BT, K]
        dec_i = decay_c[:, :, i]   # [B, HV, BT]
        L_i   = L_mask[:, :, i]    # [B, HV, BT, BT]

        # Intra-chunk attention (causal, with decay).
        attn = (q_i @ k_i.transpose(-1, -2) * L_i).masked_fill_(mask_upper_o, 0)
        # naive.py line 148-151:
        v_prime = w_i @ S                    # [B, HV, BT, V] — inter-chunk contrib
        v_new   = u_i - v_prime              # [B, HV, BT, V]
        o_inter = (q_i * dec_i.exp().unsqueeze(-1)) @ S  # [B, HV, BT, V]
        o_c[:, :, i] = o_inter + attn @ v_new

        # Update state: S = S * exp(last decay in chunk) + k^T @ v_new
        # naive.py line 152-153:
        last_dec  = dec_i[:, :, -1:]        # [B, HV, 1]
        S = (S * last_dec.exp().unsqueeze(-1)
             + (k_i * (last_dec - dec_i).exp().unsqueeze(-1)).transpose(-1, -2)
             @ v_new)

    # ---- Reshape and unpad ----------------------------------------------
    # [B, HV, NC, BT, V] → [B, HV, T_pad, V]
    o_out = o_c.view(B, HV, T_pad, V)
    # Unpad and transpose to [B, T, HV, V]
    o_out = o_out[:, :, :T, :].permute(0, 2, 1, 3)

    final_state = S if output_final_state else None
    # Cast back to v input dtype (fp16/bf16 in typical training).
    return o_out, final_state
