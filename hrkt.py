import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_kerple_bias

# ============================================================================
# Bottom Layer (Section 4.1, Eq. 5–13)
# ============================================================================

class BottomLayer(nn.Module):
    """
    Chunk-wise recurrent self-attention layer.

    Manages memory tokens (read/write) and KERPLE bias, while
    delegating the actual attention computation to the base model
    via its ``attend_chunk`` method.

    Corresponds to Section 4.1 (Eq. 5–13) in the paper.
    """

    def __init__(self, d, num_heads, chunk_len, mem_len, dropout):
        super().__init__()
        self.d = d
        self.h = num_heads
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        # Learnable memory tokens  (Section 4.1.1)
        self.read_mem_init = nn.Parameter(torch.randn(1, mem_len, d))
        self.write_mem_token = nn.Parameter(torch.randn(1, mem_len, d))

        # KERPLE parameters for token-level bias  (Eq. 8)
        self.r1 = nn.Parameter(torch.ones(num_heads) * 0.1)
        self.r2 = nn.Parameter(torch.ones(num_heads) * 0.1)

        # HRKT-specific dropout (applied within attend_chunk)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def _build_kerple(self, length, device):
        """Token-level KERPLE bias: B[i,j] = -r1·log(1 + r2·|i-j|)  (Eq. 8)"""
        pos = torch.arange(length, device=device).float()
        distances = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        return compute_kerple_bias(self.r1, self.r2, self.h, distances)

    def _build_chunk_mask(self, device):
        """Causal mask: applied only to data tokens; memory tokens unmasked."""
        C, m = self.chunk_len, self.mem_len
        T = m + C + m
        mask = torch.zeros(T, T, dtype=torch.bool, device=device)
        data_causal = torch.triu(
            torch.ones(C, C, dtype=torch.bool, device=device), diagonal=1
        )
        mask[m : m + C, m : m + C] = data_causal
        return mask

    def _forward_chunk(self, base, q_c, r_c, qry_c, read_mem, device):
        """
        Process a single chunk using the base model's attention.

        Args:
            base:     BaseKTAdapter instance.
            q_c:      Question IDs for this chunk    [B, C].
            r_c:      Responses for this chunk        [B, C].
            qry_c:    Query question IDs              [B, C].
            read_mem: Read memory from previous chunk [B, m, d].
            device:   Torch device.

        Returns:
            out_data:  Data token outputs  [B, C, d].
            out_write: Write memory output [B, m, d].
        """
        B = q_c.shape[0]
        C, m = self.chunk_len, self.mem_len
        T = m + C + m

        # Embeddings from base model
        M_data = base.embed_interaction(q_c, r_c)
        E_data = base.embed_exercise(qry_c)
        write_mem = self.write_mem_token.expand(B, -1, -1)

        # Construct augmented sequences  (Eq. 5)
        M_full = torch.cat([read_mem, M_data, write_mem], dim=1)
        E_full = torch.cat([read_mem, E_data, write_mem], dim=1)

        # Delegate attention to base model
        kerple_bias = self._build_kerple(T, device)
        attn_mask = self._build_chunk_mask(device)

        out = base.attend_chunk(
            E_full, M_full, kerple_bias, attn_mask,
            self.attn_dropout, self.ffn_dropout,
        )

        # Decompose  (Eq. 11): discard read-mem, keep data + write-mem
        out_data = out[:, m : m + C, :]
        out_write = out[:, -m:, :]

        return out_data, out_write

    def forward(self, base, q_chunks, r_chunks, qry_chunks, num_chunks, device):
        """
        Run the Bottom Layer across all chunks sequentially.

        Args:
            base:        BaseKTAdapter instance.
            q_chunks:    [B, N, C]  chunked question IDs.
            r_chunks:    [B, N, C]  chunked responses.
            qry_chunks:  [B, N, C]  chunked query question IDs.
            num_chunks:  Number of chunks N.
            device:      Torch device.

        Returns:
            H_bottom:       Concatenated data outputs  [B, N·C, d].  (Eq. 13)
            write_memories: List of N tensors, each     [B, m, d].
        """
        B = q_chunks.shape[0]
        current_read_mem = self.read_mem_init.expand(B, -1, -1)

        data_outputs = []
        write_memories = []

        for i in range(num_chunks):
            out_data, out_write = self._forward_chunk(
                base,
                q_chunks[:, i, :],
                r_chunks[:, i, :],
                qry_chunks[:, i, :],
                current_read_mem,
                device,
            )
            data_outputs.append(out_data)
            write_memories.append(out_write)
            current_read_mem = out_write  # RM_{i+1} = Z^i_WM  (Eq. 12)

        H_bottom = torch.cat(data_outputs, dim=1)  # Eq. 13
        return H_bottom, write_memories


# ============================================================================
# Memory Self-Attention (Section 4.2.2, Eq. 15–19)
# ============================================================================

class MemorySelfAttention(nn.Module):
    """
    Self-attention over collected write memories from all chunks.

    Refines the memory sequence M_collected by letting each chunk's
    summary attend to summaries from all preceding chunks, using
    chunk-level KERPLE bias and causal masking.

    Corresponds to Section 4.2.2 (Eq. 15–19) in the paper.
    """

    def __init__(self, d, num_heads, mem_len, dropout, top_dropout):
        super().__init__()
        self.d = d
        self.h = num_heads
        self.dk = d // num_heads
        self.mem_len = mem_len

        # Q, K, V projections  (Eq. 15)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

        # KERPLE parameters for chunk-level bias  (Eq. 17)
        self.r1 = nn.Parameter(torch.ones(num_heads) * 0.1)
        self.r2 = nn.Parameter(torch.ones(num_heads) * 0.1)

        # Post-attention layers  (Eq. 18–19)
        self.ln1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(top_dropout)

    def forward(self, write_memories, device):
        """
        Args:
            write_memories: List of N tensors, each [B, mem_len, d].
            device: Torch device.

        Returns:
            M_top: Refined memory tensor [B, N·mem_len, d].  (Eq. 19)
        """
        B = write_memories[0].shape[0]
        N = len(write_memories)

        if N == 0:
            return torch.zeros(B, 0, self.d, device=device)

        # M_collected = [Z^1_WM | Z^2_WM | ... | Z^N_WM]  (Eq. 14)
        mem_seq = torch.stack(write_memories, dim=1).view(B, N * self.mem_len, self.d)
        L = N * self.mem_len

        # Q, K, V  (Eq. 15)
        Q_h = self.q_proj(mem_seq).view(B, L, self.h, self.dk).transpose(1, 2)
        K_h = self.k_proj(mem_seq).view(B, L, self.h, self.dk).transpose(1, 2)
        V_h = self.v_proj(mem_seq).view(B, L, self.h, self.dk).transpose(1, 2)

        scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(self.dk)

        # Chunk-level KERPLE bias  (Eq. 17)
        token_to_chunk = torch.arange(L, device=device) // self.mem_len
        chunk_dist = (token_to_chunk.unsqueeze(1) - token_to_chunk.unsqueeze(0)).abs().float()
        scores = scores + compute_kerple_bias(self.r1, self.r2, self.h, chunk_dist).unsqueeze(0)

        # Causal mask: chunk[i] < chunk[j] → masked
        causal_mask = token_to_chunk.unsqueeze(1) < token_to_chunk.unsqueeze(0)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if torch.isnan(attn).any():
            warnings.warn("NaN in MemorySelfAttention softmax; using uniform fallback.")
            attn = torch.ones_like(attn) / attn.size(-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V_h)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d)

        # Output projection + residual + LayerNorm  (Eq. 18)
        out = self.dropout(self.o_proj(out))
        out = self.ln1(mem_seq + out)

        # FFN + residual + LayerNorm  (Eq. 19)
        out = self.ln2(out + self.ffn(out))

        return out


# ============================================================================
# Top Layer (Section 4.2.3, Eq. 20–25)
# ============================================================================

class TopLayer(nn.Module):
    """
    Cross-attention from per-timestep exercise queries to refined
    chunk-level memories, capturing long-range knowledge evolution.

    Corresponds to Section 4.2.3 (Eq. 20–25) in the paper.
    """

    def __init__(self, d, num_heads, dropout, top_dropout, top_d_ff_ratio=4):
        super().__init__()
        self.d = d
        self.h = num_heads
        self.dk = d // num_heads

        # Q from exercise embeddings (Eq. 21), K/V from refined memories (Eq. 20)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

        # KERPLE parameters for cross-attention bias  (Eq. 23)
        self.r1 = nn.Parameter(torch.ones(num_heads) * 0.1)
        self.r2 = nn.Parameter(torch.ones(num_heads) * 0.1)

        self.dropout = nn.Dropout(top_dropout)

        # Post-attention FFN + LayerNorm  (Eq. 24–25)
        top_d_ff = int(d * top_d_ff_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d, top_d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(top_d_ff, d),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, Q_seq, KV_memories, chunk_len, mem_len, L_padded, device):
        """
        Args:
            Q_seq:        Exercise query embeddings  [B, L_padded, d].
            KV_memories:  Refined memories            [B, N·mem_len, d].
            chunk_len:    Chunk size C.
            mem_len:      Memory size m.
            L_padded:     Padded sequence length.
            device:       Torch device.

        Returns:
            H_top: Top layer output  [B, L_padded, d].  (Eq. 25)
        """
        B = Q_seq.size(0)
        L_top = Q_seq.size(1)
        M_top = KV_memories.size(1)

        # Projections  (Eq. 20–21)
        Q = self.q_proj(Q_seq)
        K = self.k_proj(KV_memories)
        V = self.v_proj(KV_memories)

        Q_h = Q.view(B, L_top, self.h, self.dk).permute(0, 2, 1, 3)
        K_h = K.view(B, M_top, self.h, self.dk).permute(0, 2, 1, 3)
        V_h = V.view(B, M_top, self.h, self.dk).permute(0, 2, 1, 3)

        logits = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(self.dk)

        # Chunk-level KERPLE bias  (Eq. 23)
        q_chunk_idx = torch.arange(L_padded, device=device) // chunk_len
        k_chunk_idx = torch.arange(M_top, device=device) // mem_len
        chunk_dist = (q_chunk_idx.unsqueeze(1) - k_chunk_idx.unsqueeze(0)).abs().float()
        logits = logits + compute_kerple_bias(self.r1, self.r2, self.h, chunk_dist).unsqueeze(0)

        # Causal retrieval mask
        mask_cond = q_chunk_idx.unsqueeze(1) <= k_chunk_idx.unsqueeze(0)
        retrieval_mask = torch.zeros((L_padded, M_top), device=device)
        retrieval_mask = retrieval_mask.masked_fill(mask_cond, float("-inf"))
        logits = logits + retrieval_mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(logits, dim=-1)
        if torch.isnan(attn).any():
            warnings.warn("NaN in TopLayer cross-attention softmax; using uniform fallback.")
            attn = torch.ones_like(attn) / attn.size(-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V_h)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L_top, self.d)

        # Output projection + residual + LayerNorm  (Eq. 24)
        out = self.dropout(self.o_proj(out))
        out = self.ln1(Q_seq + out)

        # FFN + residual + LayerNorm  (Eq. 25)
        out = self.ln2(out + self.ffn(out))

        return out


# ============================================================================
# Fusion Gate (Section 4.3, Eq. 26–27)
# ============================================================================

class FusionGate(nn.Module):
    """
    Gated fusion of Bottom Layer (local) and Top Layer (global) outputs.

        gate    = σ([H_bottom ‖ H_top] · W_g)                   (Eq. 26)
        H_fused = LayerNorm(H_bottom + Dropout(gate ⊙ H_top))   (Eq. 27)

    Corresponds to Section 4.3 in the paper.
    """

    def __init__(self, d, top_dropout):
        super().__init__()
        self.gate_proj = nn.Linear(d * 2, d)  # W_g
        self.ln = nn.LayerNorm(d)
        self.dropout = nn.Dropout(top_dropout)

    def forward(self, H_bottom, H_top):
        """
        Args:
            H_bottom: Bottom layer output  [B, L, d].
            H_top:    Top layer output      [B, L, d].

        Returns:
            H_fused:  Fused representation  [B, L, d].
        """
        concat = torch.cat([H_bottom, H_top], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))     # Eq. 26
        fused = H_bottom + self.dropout(gate * H_top)    # Eq. 27
        return self.ln(fused)


# ============================================================================
# Prediction Head (Section 4.4, Eq. 28)
# ============================================================================

class PredictionHead(nn.Module):
    """
    Two-layer FFN with ReLU and dropout, followed by sigmoid.

        r̂_t = σ(W₂ · Dropout(ReLU(W₁ · H_t + b₁)) + b₂)   (Eq. 28)

    Corresponds to Section 4.4 in the paper.
    """

    def __init__(self, d, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

    def forward(self, H_fused):
        """
        Args:
            H_fused: Fused representation  [B, L, d].

        Returns:
            Predicted correctness probabilities  [B, L].
        """
        return torch.sigmoid(self.ffn(H_fused).squeeze(-1))


# ============================================================================
# HRKT Framework (Section 4)
# ============================================================================

class HRKT(nn.Module):
    """
    Hierarchical Recurrent Knowledge Tracing — a model-agnostic framework.

    Wraps any BaseKTAdapter with a hierarchical recurrent structure:
    BottomLayer → MemorySelfAttention → TopLayer → FusionGate → PredictionHead.

    Usage::

        sakt = SAKTAdapter(num_q=100, d=64, num_heads=4, dropout=0.2)
        model = HRKT(sakt, chunk_len=50, mem_len=4)

        probs = model(q, r, t, qry)  # [B, L]

    Args:
        base_model:      BaseKTAdapter instance (e.g., SAKTAdapter).
        chunk_len:       Chunk size C (default 50).
        mem_len:         Memory token count m per chunk (default 4).
        dropout:         Base dropout rate (default 0.2).
        top_dropout:     Dropout rate for top layer and fusion (default 0.3).
        top_d_ff_ratio:  FFN expansion ratio for top layer (default 4).
    """

    def __init__(
        self,
        base_model,
        chunk_len=50,
        mem_len=4,
        dropout=0.2,
        top_dropout=0.3,
        top_d_ff_ratio=4,
    ):
        super().__init__()
        d = base_model.d
        num_heads = base_model.num_heads

        self.base = base_model
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        self.bottom = BottomLayer(d, num_heads, chunk_len, mem_len, dropout)
        self.memory_self_attn = MemorySelfAttention(d, num_heads, mem_len, dropout, top_dropout)
        self.top_layer = TopLayer(d, num_heads, dropout, top_dropout, top_d_ff_ratio)
        self.fusion_gate = FusionGate(d, top_dropout)
        self.prediction_head = PredictionHead(d, dropout)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _pad_to_chunk_len(tensor, chunk_len, device):
        """Pad a [B, L] tensor so L becomes a multiple of chunk_len."""
        remainder = tensor.size(1) % chunk_len
        if remainder != 0:
            pad_len = chunk_len - remainder
            pad = torch.zeros(tensor.size(0), pad_len, dtype=tensor.dtype, device=device)
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor

    def forward(self, q, r, t, qry):
        """
        Full HRKT forward pass.
        """
        B, L = q.shape
        device = q.device

        # 1. Pad & chunk
        q = self._pad_to_chunk_len(q, self.chunk_len, device)
        r = self._pad_to_chunk_len(r, self.chunk_len, device)
        qry = self._pad_to_chunk_len(qry, self.chunk_len, device)

        L_padded = q.shape[1]
        num_chunks = L_padded // self.chunk_len

        if num_chunks == 0:
            return torch.zeros(B, L, device=device)

        q_chunks = q.reshape(B, num_chunks, self.chunk_len)
        r_chunks = r.reshape(B, num_chunks, self.chunk_len)
        qry_chunks = qry.reshape(B, num_chunks, self.chunk_len)

        # 2. Bottom Layer  (Section 4.1)
        H_bottom, write_memories = self.bottom(
            self.base, q_chunks, r_chunks, qry_chunks, num_chunks, device,
        )

        # 3. Memory Self-Attention  (Section 4.2.2)
        refined_memories = self.memory_self_attn(write_memories, device)

        # 4. Top Layer  (Section 4.2.3)
        Q_seq = self.base.embed_exercise(qry)
        if Q_seq.size(1) < L_padded:
            pad = L_padded - Q_seq.size(1)
            Q_seq = torch.cat(
                [Q_seq, torch.zeros(B, pad, self.base.d, device=device)], dim=1
            )

        if refined_memories.size(1) == 0:
            H_fused = self.fusion_gate.ln(H_bottom)
            return self.prediction_head(H_fused)[:, :L]

        H_top = self.top_layer(
            Q_seq, refined_memories,
            self.chunk_len, self.mem_len, L_padded, device,
        )

        # 5. Fusion  (Section 4.3)
        H_fused = self.fusion_gate(H_bottom, H_top)

        # 6. Prediction  (Section 4.4)
        probs = self.prediction_head(H_fused)

        return probs[:, :L]
