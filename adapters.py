import math
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Base KT Adapter Interface
# ============================================================================

class BaseKTAdapter(ABC, nn.Module):
    """
    Interface for base KT models that can be plugged into HRKT.

    Any attention-based KT model can be integrated by implementing
    this interface. See SAKTAdapter for a reference implementation.

    Required properties:
        d:         Hidden dimension.
        num_heads: Number of attention heads.

    Required methods:
        embed_exercise(q)        → exercise embeddings   [B, L, d]
        embed_interaction(q, r)  → interaction embeddings [B, L, d]
        attend_chunk(E_full, M_full, kerple_bias, attn_mask,
                     attn_dropout, ffn_dropout)
                                 → attention output       [B, T, d]
    """

    @property
    @abstractmethod
    def d(self):
        """Hidden dimension."""

    @property
    @abstractmethod
    def num_heads(self):
        """Number of attention heads."""

    @abstractmethod
    def embed_exercise(self, q):
        """
        Map question IDs to exercise embeddings.

        Args:
            q: Question IDs [B, L].
        Returns:
            Exercise embeddings [B, L, d].
        """

    @abstractmethod
    def embed_interaction(self, q, r):
        """
        Map (question, response) pairs to interaction embeddings.

        Args:
            q: Question IDs [B, L].
            r: Response labels (0 or 1) [B, L].
        Returns:
            Interaction embeddings [B, L, d].
        """

    @abstractmethod
    def attend_chunk(self, E_full, M_full, attention_bias, attn_mask,
                     attn_dropout=None, ffn_dropout=None):
        """
        Apply self-attention + FFN within a single chunk.

        The base model always receives an attention mask.
        HRKT can optionally inject an additional attention bias such as KERPLE.

        Args:
            E_full:       [B, T, d]  exercise embeddings (→ Q, residual).
            M_full:       [B, T, d]  interaction embeddings (→ K, V).
            attention_bias:  Optional attention bias [num_heads, T, T].
                             For HRKT, this is KERPLE bias.
                             If None, the model behaves like the base attention.
            attn_mask:    [T, T]  bool mask (True → masked).
            attn_dropout: Optional nn.Dropout for attention weights.
            ffn_dropout:  Optional nn.Dropout for FFN output.

        Returns:
            Output tensor [B, T, d].
        """


# ============================================================================
# SAKT Adapter (Reference Implementation)
# ============================================================================

class SAKTAdapter(BaseKTAdapter):
    """
    SAKT (Self-Attentive Knowledge Tracing) adapter for HRKT.
    """

    def __init__(self, num_q, d, num_heads, dropout):
        super().__init__()
        self.num_q = num_q
        self._d = d
        self._num_heads = num_heads
        self._dk = d // num_heads

        # Embeddings (Eq. 3–4)
        self.E = nn.Embedding(num_q + 1, d, padding_idx=0)      # exercise
        self.M = nn.Embedding(2 * num_q + 1, d, padding_idx=0)  # interaction

        # Multi-head attention projections (Eq. 6)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)

        # Post-attention layers (Eq. 9–10)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.Dropout(dropout),
        )

    @property
    def d(self):
        return self._d

    @property
    def num_heads(self):
        return self._num_heads

    def embed_exercise(self, q):
        return self.E(q)

    def embed_interaction(self, q, r):
        x = q + self.num_q * r
        return self.M(x)

    def attend_chunk(self, E_full, M_full, attention_bias, attn_mask,
                     attn_dropout=None, ffn_dropout=None):
        B, T, _ = E_full.shape
        h, dk = self._num_heads, self._dk

        # Q from exercise, K/V from interaction  (Eq. 6)
        Q = self.W_Q(E_full).view(B, T, h, dk).permute(0, 2, 1, 3)
        K = self.W_K(M_full).view(B, T, h, dk).permute(0, 2, 1, 3)
        V = self.W_V(M_full).view(B, T, h, dk).permute(0, 2, 1, 3)

        # Scaled dot-product attention  
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        # HRKT injects KERPLE here. (Eq. 7)
        # If attention_bias is None, this remains base attention.
        if attention_bias is not None:
            logits = logits + attention_bias.unsqueeze(0)

        logits = logits.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(logits, dim=-1)
        if torch.isnan(attn).any():
            warnings.warn("NaN in attend_chunk softmax; using uniform fallback.")
            attn = torch.ones_like(attn) / attn.size(-1)
        if attn_dropout is not None:
            attn = attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, T, self._d)

        # Output projection + residual + LayerNorm + FFN  (Eq. 9–10)
        proj = self.W_O(out)
        if attn_dropout is not None:
            proj = attn_dropout(proj)
        out = self.ln1(E_full + proj)

        ffn_out = self.ffn(out)
        if ffn_dropout is not None:
            ffn_out = ffn_dropout(ffn_out)
        out = self.ln2(out + ffn_out)

        return out
