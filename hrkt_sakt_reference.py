"""
HRKT-SAKT reference implementation.

This file contains a compact, modular implementation of the HRKT-SAKT model:
1) Bottom layer: chunk-wise recurrent self-attention with read/write memory tokens.
2) Top layer: memory self-attention + cross-attention from query tokens to chunk memories.
3) Fusion layer: gated fusion of local bottom representations and global top representations.

The implementation is intentionally written as a single-file example for GitHub release.
It preserves the behavior of the provided experimental code, including:
- question embedding E(qry)
- interaction embedding M(q + num_q * r)
- KERPLE-style logarithmic relative bias
- bottom-layer recurrent write-memory handoff
- top-layer memory self-attention
- causal retrieval mask that blocks current/future chunk memories
- gated fusion and final two-layer FFN prediction
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KERPLELogBias(nn.Module):
    """Head-wise logarithmic KERPLE relative position bias.

    Bias formula:
        B[i, j] = -softplus(r1_h) * log(1 + softplus(r2_h) * distance(i, j))
    """

    def __init__(self, num_heads: int, init_value: float = 0.1, max_r: float = 10.0):
        super().__init__()
        self.num_heads = num_heads
        self.max_r = max_r
        self.r1 = nn.Parameter(torch.ones(num_heads) * init_value)
        self.r2 = nn.Parameter(torch.ones(num_heads) * init_value)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Return bias with shape [num_heads, *distance.shape].

        Args:
            distance: non-negative distance tensor, e.g. [L, L] or [Lq, Lk].
        """
        r1 = torch.clamp(F.softplus(self.r1), max=self.max_r)
        r2 = torch.clamp(F.softplus(self.r2), max=self.max_r)

        while r1.dim() < distance.dim() + 1:
            r1 = r1.unsqueeze(-1)
            r2 = r2.unsqueeze(-1)

        return -r1 * torch.log1p(torch.clamp(r2 * distance.unsqueeze(0), max=1e4))


class PositionwiseFFN(nn.Module):
    """Two-layer feed-forward network used inside Transformer blocks."""

    def __init__(self, d_model: int, dropout: float, expansion: int = 1):
        super().__init__()
        d_hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[B, L, D] -> [B, H, L, Dh]."""
    bsz, seq_len, dim = x.shape
    head_dim = dim // num_heads
    return x.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, H, L, Dh] -> [B, L, D]."""
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)


class HRKTSAKTBottomLayer(nn.Module):
    """Bottom layer: chunk-wise recurrent memory Transformer.

    For each chunk, this layer builds:
        [read_memory | interaction/question tokens | write_memory]

    It returns:
        bottom_output: timestep-wise local representation, [B, L_padded, D]
        write_memories: list of chunk summary memories, each [B, mem_len, D]
    """

    def __init__(
        self,
        num_q: int,
        d_model: int,
        num_heads: int,
        dropout: float,
        chunk_len: int,
        mem_len: int,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.num_q = num_q
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        # SAKT-style embeddings
        self.question_embed = nn.Embedding(num_q + 1, d_model, padding_idx=0)
        self.interaction_embed = nn.Embedding(2 * num_q + 1, d_model, padding_idx=0)

        # RMT memory tokens
        self.read_mem_init = nn.Parameter(torch.randn(1, mem_len, d_model) * 0.02)
        self.write_mem_token = nn.Parameter(torch.randn(1, mem_len, d_model) * 0.02)

        # Kept for architectural clarity and checkpoint extensibility.
        # The original experimental forward path did not add this parameter.
        self.pos_bottom = nn.Parameter(torch.empty(chunk_len + 2 * mem_len, d_model))
        nn.init.xavier_uniform_(self.pos_bottom)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.kerple = KERPLELogBias(num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, dropout, expansion=1)
        self.ln2 = nn.LayerNorm(d_model)

    def _build_bottom_mask(self, total_len: int, device: torch.device) -> torch.Tensor:
        """Causal mask only among data tokens; memory tokens remain unmasked."""
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

        data_start = self.mem_len
        data_end = self.mem_len + self.chunk_len
        data_causal = torch.triu(
            torch.ones(self.chunk_len, self.chunk_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask[data_start:data_end, data_start:data_end] = data_causal
        return mask

    def _kerple_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device).float()
        distance = (pos[None, :] - pos[:, None]).abs()
        return self.kerple(distance)  # [H, T, T]

    def forward(
        self,
        q_chunks: torch.Tensor,
        r_chunks: torch.Tensor,
        qry_chunks: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bsz, num_chunks, _ = q_chunks.shape
        device = q_chunks.device
        total_len = self.mem_len + self.chunk_len + self.mem_len

        current_read_mem = self.read_mem_init.expand(bsz, -1, -1)
        bottom_outputs = []
        write_memories = []

        attn_mask = self._build_bottom_mask(total_len, device)
        kerple_bias = self._kerple_bias(total_len, device)

        for chunk_idx in range(num_chunks):
            q_c = q_chunks[:, chunk_idx, :]
            r_c = r_chunks[:, chunk_idx, :]
            qry_c = qry_chunks[:, chunk_idx, :]

            interaction_idx = q_c + self.num_q * r_c
            m_data = self.interaction_embed(interaction_idx)
            e_data = self.question_embed(qry_c)

            read_mem = current_read_mem
            write_mem = self.write_mem_token.expand(bsz, -1, -1)

            m_full = torch.cat([read_mem, m_data, write_mem], dim=1)
            e_full = torch.cat([read_mem, e_data, write_mem], dim=1)

            q_proj = split_heads(self.q_proj(e_full), self.num_heads)
            k_proj = split_heads(self.k_proj(m_full), self.num_heads)
            v_proj = split_heads(self.v_proj(m_full), self.num_heads)

            logits = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
            logits = logits + kerple_bias.unsqueeze(0)
            logits = logits.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = F.softmax(logits, dim=-1)
            if torch.isnan(attn).any():
                attn = torch.ones_like(attn) / attn.size(-1)
            attn = self.attn_dropout(attn)

            out = torch.matmul(attn, v_proj)
            out = merge_heads(out)

            attn_out = self.attn_dropout(self.o_proj(out))
            out = self.ln1(e_full + attn_out)

            ffn_out = self.ffn_dropout(self.ffn(out))
            out = self.ln2(out + ffn_out)

            out_data = out[:, self.mem_len : self.mem_len + self.chunk_len, :]
            out_write = out[:, -self.mem_len :, :]

            bottom_outputs.append(out_data)
            write_memories.append(out_write)
            current_read_mem = out_write

        bottom_output = torch.cat(bottom_outputs, dim=1)
        return bottom_output, write_memories


class HRKTSAKTTopLayer(nn.Module):
    """Top layer: memory self-attention followed by cross-attention.

    1) Refine collected write memories with causal chunk-level memory self-attention.
    2) Let each query timestep attend to refined memories from previous chunks only.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        top_dropout: float,
        chunk_len: int,
        mem_len: int,
        top_d_ff_ratio: int = 4,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        # Memory self-attention
        self.mem_q_proj = nn.Linear(d_model, d_model)
        self.mem_k_proj = nn.Linear(d_model, d_model)
        self.mem_v_proj = nn.Linear(d_model, d_model)
        self.mem_o_proj = nn.Linear(d_model, d_model)
        self.mem_kerple = KERPLELogBias(num_heads)
        self.mem_ln1 = nn.LayerNorm(d_model)
        self.mem_ffn = PositionwiseFFN(d_model, dropout, expansion=2)
        self.mem_ln2 = nn.LayerNorm(d_model)
        self.mem_dropout = nn.Dropout(top_dropout)

        # Top cross-attention
        self.top_q_proj = nn.Linear(d_model, d_model)
        self.top_k_proj = nn.Linear(d_model, d_model)
        self.top_v_proj = nn.Linear(d_model, d_model)
        self.top_o_proj = nn.Linear(d_model, d_model)
        self.top_kerple = KERPLELogBias(num_heads)
        self.top_dropout = nn.Dropout(top_dropout)
        self.top_ln1 = nn.LayerNorm(d_model)
        self.top_ffn = PositionwiseFFN(d_model, dropout, expansion=top_d_ff_ratio)
        self.top_ln2 = nn.LayerNorm(d_model)

        self._init_projection_weights()

    def _init_projection_weights(self) -> None:
        for module in [
            self.mem_q_proj, self.mem_k_proj, self.mem_v_proj, self.mem_o_proj,
            self.top_q_proj, self.top_k_proj, self.top_v_proj, self.top_o_proj,
        ]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def memory_self_attention(
        self,
        write_memories: List[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if len(write_memories) == 0:
            # This case is rarely reached because forward returns early when num_chunks == 0.
            return torch.zeros(0, 0, self.d_model, device=device)

        bsz = write_memories[0].shape[0]
        num_chunks = len(write_memories)
        mem_seq = torch.stack(write_memories, dim=1).view(
            bsz, num_chunks * self.mem_len, self.d_model
        )

        mem_len_total = mem_seq.size(1)

        q_mem = split_heads(self.mem_q_proj(mem_seq), self.num_heads)
        k_mem = split_heads(self.mem_k_proj(mem_seq), self.num_heads)
        v_mem = split_heads(self.mem_v_proj(mem_seq), self.num_heads)

        scores = torch.matmul(q_mem, k_mem.transpose(-2, -1)) / math.sqrt(self.head_dim)

        token_to_chunk = torch.arange(mem_len_total, device=device) // self.mem_len
        chunk_distance = (token_to_chunk[:, None] - token_to_chunk[None, :]).abs().float()
        scores = scores + self.mem_kerple(chunk_distance).unsqueeze(0)

        # Causal over chunks: a memory token cannot attend to future chunk memories.
        causal_mask = token_to_chunk[:, None] < token_to_chunk[None, :]
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.ones_like(attn) / attn.size(-1)
        attn = self.mem_dropout(attn)

        out = torch.matmul(attn, v_mem)
        out = merge_heads(out)

        out = self.mem_dropout(self.mem_o_proj(out))
        out = self.mem_ln1(mem_seq + out)

        ffn_out = self.mem_ffn(out)
        out = self.mem_ln2(out + ffn_out)

        return out

    def forward(
        self,
        q_seq: torch.Tensor,
        write_memories: List[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = q_seq.shape
        device = q_seq.device

        refined_mem = self.memory_self_attention(write_memories, device)
        if refined_mem.size(1) == 0:
            return torch.zeros_like(q_seq)

        q_top = self.top_q_proj(q_seq)
        k_top = self.top_k_proj(refined_mem)
        v_top = self.top_v_proj(refined_mem)

        q_h = split_heads(q_top, self.num_heads)
        k_h = split_heads(k_top, self.num_heads)
        v_h = split_heads(v_top, self.num_heads)

        logits = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)

        num_mem_tokens = k_top.size(1)
        q_chunk = torch.arange(seq_len, device=device) // self.chunk_len
        k_chunk = torch.arange(num_mem_tokens, device=device) // self.mem_len

        chunk_distance = (q_chunk[:, None] - k_chunk[None, :]).abs().float()
        logits = logits + self.top_kerple(chunk_distance).unsqueeze(0)

        # Causal retrieval: query at chunk c can only retrieve memories from chunks < c.
        retrieval_mask = q_chunk[:, None] <= k_chunk[None, :]
        logits = logits.masked_fill(retrieval_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(logits, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.ones_like(attn) / attn.size(-1)
        attn = self.top_dropout(attn)

        top_out = torch.matmul(attn, v_h)
        top_out = merge_heads(top_out)

        top_attn_out = self.top_dropout(self.top_o_proj(top_out))
        top_out = self.top_ln1(q_seq + top_attn_out)

        top_ffn_out = self.top_ffn(top_out)
        top_out = self.top_ln2(top_out + top_ffn_out)

        return top_out


class HRKTFusionLayer(nn.Module):
    """Gated fusion of bottom local features and top global features."""

    def __init__(self, d_model: int, dropout: float, top_dropout: float):
        super().__init__()
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.fusion_ln = nn.LayerNorm(d_model)
        self.top_dropout = nn.Dropout(top_dropout)
        self.prediction = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, bottom_output: torch.Tensor, top_output: torch.Tensor) -> torch.Tensor:
        concat_features = torch.cat([bottom_output, top_output], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(concat_features))
        fused = bottom_output + self.top_dropout(gate * top_output)
        fused = self.fusion_ln(fused)

        logits = self.prediction(fused).squeeze(-1)
        return torch.sigmoid(logits)


class HRKT_SAKT(nn.Module):
    """Hierarchical Recurrent Knowledge Tracing based on SAKT.

    Args:
        num_q: number of unique questions/items. Index 0 is reserved for padding.
        max_len: kept for compatibility with existing training scripts.
        d_model: hidden dimension.
        num_heads: number of attention heads.
        dropout: dropout used inside attention/FFN blocks.
        chunk_len: number of timesteps per chunk.
        mem_len: number of memory tokens per chunk.
        top_dropout: dropout used for top attention/fusion.
        top_d_ff_ratio: expansion ratio of top-layer FFN.
    """

    def __init__(
        self,
        num_q: int,
        max_len: int,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
        chunk_len: int = 50,
        mem_len: int = 4,
        top_dropout: float = 0.3,
        top_d_ff_ratio: int = 4,
    ):
        super().__init__()
        self.num_q = num_q
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        self.bottom = HRKTSAKTBottomLayer(
            num_q=num_q,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            chunk_len=chunk_len,
            mem_len=mem_len,
        )
        self.top = HRKTSAKTTopLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            top_dropout=top_dropout,
            chunk_len=chunk_len,
            mem_len=mem_len,
            top_d_ff_ratio=top_d_ff_ratio,
        )
        self.fusion = HRKTFusionLayer(
            d_model=d_model,
            dropout=dropout,
            top_dropout=top_dropout,
        )

        self._init_weights()

    @property
    def question_embed(self) -> nn.Embedding:
        """Expose question embedding for compatibility/readability."""
        return self.bottom.question_embed

    @property
    def interaction_embed(self) -> nn.Embedding:
        """Expose interaction embedding for compatibility/readability."""
        return self.bottom.interaction_embed

    def _init_weights(self) -> None:
        """Match the original broad initialization policy."""
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
    def _pad_to_chunk(
        q: torch.Tensor,
        r: torch.Tensor,
        qry: torch.Tensor,
        chunk_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Pad sequence length to a multiple of chunk_len."""
        bsz, seq_len = q.shape
        remainder = seq_len % chunk_len
        pad_len = 0 if remainder == 0 else chunk_len - remainder

        if pad_len > 0:
            pad_q = torch.zeros(bsz, pad_len, dtype=torch.long, device=q.device)
            pad_r = torch.zeros(bsz, pad_len, dtype=torch.long, device=r.device)
            pad_qry = torch.zeros(bsz, pad_len, dtype=torch.long, device=qry.device)
            q = torch.cat([q, pad_q], dim=1)
            r = torch.cat([r, pad_r], dim=1)
            qry = torch.cat([qry, pad_qry], dim=1)

        return q, r, qry, pad_len

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        qry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict correctness probabilities.

        Args:
            q: previous question IDs, [B, L], values in [0, num_q].
            r: previous correctness labels, [B, L], values in {0, 1}.
            t: optional timestamp/interval tensor. Kept for API compatibility; not used.
            qry: target/query question IDs, [B, L]. If omitted, q is used.

        Returns:
            probabilities, [B, original_L].
        """
        del t  # API compatibility with KT pipelines that pass time features.

        if qry is None:
            qry = q

        if q.dtype != torch.long or r.dtype != torch.long or qry.dtype != torch.long:
            raise TypeError("q, r, and qry must be torch.long tensors.")

        original_len = q.size(1)

        assert q.min() >= 0 and q.max() <= self.num_q
        assert r.min() >= 0 and r.max() <= 1
        assert qry.min() >= 0 and qry.max() <= self.num_q

        q, r, qry, _ = self._pad_to_chunk(q, r, qry, self.chunk_len)
        padded_len = q.size(1)
        num_chunks = padded_len // self.chunk_len

        if num_chunks == 0:
            return torch.zeros(q.size(0), original_len, device=q.device)

        q_chunks = q.view(q.size(0), num_chunks, self.chunk_len)
        r_chunks = r.view(r.size(0), num_chunks, self.chunk_len)
        qry_chunks = qry.view(qry.size(0), num_chunks, self.chunk_len)

        bottom_output, write_memories = self.bottom(q_chunks, r_chunks, qry_chunks)

        q_seq = self.question_embed(qry)
        top_output = self.top(q_seq, write_memories)

        probs = self.fusion(bottom_output, top_output)
        return probs[:, :original_len]


def random_kt_batch(
    batch_size: int,
    seq_len: int,
    num_q: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dummy KT tensors for a shape check."""
    q = torch.randint(1, num_q + 1, (batch_size, seq_len), device=device)
    r = torch.randint(0, 2, (batch_size, seq_len), device=device)
    qry = torch.randint(1, num_q + 1, (batch_size, seq_len), device=device)
    return q.long(), r.long(), qry.long()


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 64
    seq_len = 1000
    num_q = 1200

    q, r, qry = random_kt_batch(batch_size, seq_len, num_q)

    model = HRKT_SAKT(
        num_q=num_q,
        max_len=seq_len,
        d_model=128,
        num_heads=8,
        dropout=0.2,
        chunk_len=50,
        mem_len=4,
        top_dropout=0.3,
        top_d_ff_ratio=4,
    )

    with torch.no_grad():
        out = model(q=q, r=r, qry=qry)

    print("Output shape:", out.shape)  # expected: torch.Size([64, 1000])
