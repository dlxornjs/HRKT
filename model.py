import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================================================
# 1. Base SAKT Model (Original Standalone SAKT Model)
# ================================================================================================
class SAKT(nn.Module):
    def __init__(self, num_q, max_len, d, num_heads, dropout):
        super().__init__()
        self.num_q = num_q
        self.max_len = max_len
        self.d = d
        self.h = num_heads
        self.dk = d // num_heads

        self.E = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.M = nn.Embedding(2 * num_q + 1, d, padding_idx=0)
        self.P = nn.Parameter(torch.Tensor(max_len + 1, d))
        nn.init.kaiming_normal_(self.P)

        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.pred = nn.Linear(d, 1)
        
        self.r1 = nn.Parameter(torch.ones(num_heads) * 0.1)
        self.r2 = nn.Parameter(torch.ones(num_heads) * 0.1)

    def build_kerple_log(self, L, device):
        i = torch.arange(L, device=device).float().unsqueeze(0)
        j = torch.arange(L, device=device).float().unsqueeze(1)
        dist = (i - j).abs()
        r1 = F.softplus(self.r1)
        r2 = F.softplus(self.r2)
        
        r1 = torch.clamp(r1, max=10.0)
        r2 = torch.clamp(r2, max=10.0)
        
        bias = -r1.view(self.h, 1, 1) * torch.log1p(
            torch.clamp(r2.view(self.h, 1, 1) * dist.unsqueeze(0), max=1e4)
        )
        return bias

    def forward(self, q, r, t, qry, return_attn=False):
        B, L = q.shape
        x = q + self.num_q * r
        M = self.M(x)
        E = self.E(qry)

        pos = self.P[:L]
        if pos.size(0) < L:
            extra_len = L - self.P.size(0)
            extra_pos = torch.zeros(extra_len, self.d, device=self.P.device)
            extended_pos = torch.cat([self.P, extra_pos], dim=0)
            pos = extended_pos[:L]
        pos = pos.unsqueeze(0)

        E = E + pos
        M = M + pos

        Q = self.W_Q(E)
        K = self.W_K(M)
        V = self.W_V(M)

        def split_heads(x):
            return x.view(B, L, self.h, self.dk).permute(0, 2, 1, 3)

        Q_h = split_heads(Q)
        K_h = split_heads(K)
        V_h = split_heads(V)

        raw = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(self.dk)

        L_idx = raw.size(-1)
        causal = torch.triu(
            torch.ones(L_idx, L_idx, device=raw.device, dtype=torch.bool),
            diagonal=1
        )
        raw = raw.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        raw = raw + self.build_kerple_log(L_idx, raw.device).unsqueeze(0)

        attn = F.softmax(raw, dim=-1)
        out = torch.matmul(attn, V_h)
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.d)
        out = self.W_O(out)

        out = self.ln1(out + E)
        out = self.ln2(out + self.ffn(out))

        logits = self.pred(out).squeeze(-1)
        probs = torch.sigmoid(logits)

        if return_attn:
            return probs, attn
        return probs

# ================================================================================================
# 2. HRKT Sub-Modules (Bottom, Top, Fusion Layers)
# ================================================================================================

class HRKT_BottomLayer(nn.Module):
    """
    Chunk-level local information processing module based on Recurrent Memory Transformer (RMT)
    """
    def __init__(self, d, num_heads, dropout, chunk_len, mem_len):
        super().__init__()
        self.d = d
        self.h = num_heads
        self.dk = d // num_heads
        self.chunk_len = chunk_len
        self.mem_len = mem_len

        self.read_mem_init = nn.Parameter(torch.randn(1, mem_len, d) * 0.02)
        self.write_mem_token = nn.Parameter(torch.randn(1, mem_len, d) * 0.02)

        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)

        self.r1 = nn.Parameter(torch.ones(num_heads) * 0.1)
        self.r2 = nn.Parameter(torch.ones(num_heads) * 0.1)

        self.attn_dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d)
        
        self.ffn = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.Dropout(dropout),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d)

    def build_kerple_log(self, L, device):
        i = torch.arange(L, device=device).float().unsqueeze(0)
        j = torch.arange(L, device=device).float().unsqueeze(1)
        dist = (i - j).abs()
        r1 = F.softplus(self.r1)
        r2 = F.softplus(self.r2)
        
        r1 = torch.clamp(r1, max=10.0)
        r2 = torch.clamp(r2, max=10.0)
        
        bias = -r1.view(self.h, 1, 1) * torch.log1p(
            torch.clamp(r2.view(self.h, 1, 1) * dist.unsqueeze(0), max=1e4)
        )
        return bias

    def forward(self, q_chunks, r_chunks, qry_chunks, E_emb, M_emb, num_q):
        B, num_chunks, _ = q_chunks.shape
        device = q_chunks.device
        
        current_read_mem = self.read_mem_init.expand(B, -1, -1)
        bottom_outputs_data = []
        all_write_memories = []

        for i in range(num_chunks):
            q_c = q_chunks[:, i, :]
            r_c = r_chunks[:, i, :]
            qry_c = qry_chunks[:, i, :]

            x = q_c + num_q * r_c
            M_data = M_emb(x)
            E_data = E_emb(qry_c)

            read_mem = current_read_mem
            write_mem = self.write_mem_token.expand(B, -1, -1)

            M_full = torch.cat([read_mem, M_data, write_mem], dim=1)
            E_full = torch.cat([read_mem, E_data, write_mem], dim=1)

            T = self.mem_len + self.chunk_len + self.mem_len

            Q = self.W_Q(E_full)
            K = self.W_K(M_full)
            V = self.W_V(M_full)

            def split_heads(x):
                return x.view(B, T, self.h, self.dk).permute(0, 2, 1, 3)

            Q_h = split_heads(Q)
            K_h = split_heads(K)
            V_h = split_heads(V)

            raw_logits = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(self.dk)
            kerple_bias = self.build_kerple_log(T, device)
            logits = raw_logits + kerple_bias.unsqueeze(0)

            # Causal masking for data tokens
            mask = torch.zeros(T, T, dtype=torch.bool, device=device)
            data_start = self.mem_len
            data_end = self.mem_len + self.chunk_len
            data_causal = torch.triu(
                torch.ones(self.chunk_len, self.chunk_len, dtype=torch.bool, device=device),
                diagonal=1
            )
            mask[data_start:data_end, data_start:data_end] = data_causal
            logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn = F.softmax(logits, dim=-1)
            if torch.isnan(attn).any():
                attn = torch.ones_like(attn) / attn.size(-1)
            attn = self.attn_dropout(attn)

            out = torch.matmul(attn, V_h)
            out = out.permute(0, 2, 1, 3).reshape(B, T, self.d)

            attn_out = self.W_O(out)
            attn_out = self.attn_dropout(attn_out)
            out = self.ln1(E_full + attn_out)

            ffn_out = self.ffn(out)
            ffn_out = self.ffn_dropout(ffn_out)
            out = self.ln2(out + ffn_out)

            out_data = out[:, self.mem_len : self.mem_len + self.chunk_len, :]
            out_write = out[:, -self.mem_len :, :]

            bottom_outputs_data.append(out_data)
            all_write_memories.append(out_write)
            current_read_mem = out_write

        bottom_output = torch.cat(bottom_outputs_data, dim=1)
        return bottom_output, all_write_memories


class HRKT_TopLayer(nn.Module):
    """
    Global Context processing module based on chunk-level summary memory
    """
    def __init__(self, d, num_heads, dropout, top_dropout, mem_len, chunk_len, top_d_ff_ratio):
        super().__init__()
        self.d = d
        self.h = num_heads
        self.dk = d // num_heads
        self.mem_len = mem_len
        self.chunk_len = chunk_len

        # Memory Self-Attention
        self.mem_self_attn_q = nn.Linear(d, d)
        self.mem_self_attn_k = nn.Linear(d, d)
        self.mem_self_attn_v = nn.Linear(d, d)
        self.mem_self_attn_o = nn.Linear(d, d)
        
        self.r1_mem_self = nn.Parameter(torch.ones(num_heads) * 0.1)
        self.r2_mem_self = nn.Parameter(torch.ones(num_heads) * 0.1)
        
        self.mem_self_ln1 = nn.LayerNorm(d)
        self.mem_self_ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.Dropout(dropout),
        )
        self.mem_self_ln2 = nn.LayerNorm(d)
        self.mem_self_dropout = nn.Dropout(top_dropout)

        # Top Layer Cross-Attention
        self.top_q_proj = nn.Linear(d, d)
        self.top_k_proj = nn.Linear(d, d)
        self.top_v_proj = nn.Linear(d, d)
        self.top_o_proj = nn.Linear(d, d)
        
        self.r1_top = nn.Parameter(torch.ones(num_heads) * 0.1)
        self.r2_top = nn.Parameter(torch.ones(num_heads) * 0.1)

        self.top_dropout = nn.Dropout(top_dropout)

        top_d_ff = int(d * top_d_ff_ratio)
        self.top_ffn = nn.Sequential(
            nn.Linear(d, top_d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(top_d_ff, d),
            nn.Dropout(dropout),
        )
        self.top_ln1 = nn.LayerNorm(d)
        self.top_ln2 = nn.LayerNorm(d)

    def memory_self_attention(self, write_memories, device):
        B = write_memories[0].shape[0]
        N = len(write_memories)
        
        if N == 0:
            return torch.zeros(B, 0, self.d, device=device)
        
        mem_seq = torch.stack(write_memories, dim=1) 
        mem_seq = mem_seq.view(B, N * self.mem_len, self.d)
        
        Q = self.mem_self_attn_q(mem_seq)
        K = self.mem_self_attn_k(mem_seq)
        V = self.mem_self_attn_v(mem_seq)
        
        L = N * self.mem_len
        Q_h = Q.view(B, L, self.h, self.dk).transpose(1, 2)
        K_h = K.view(B, L, self.h, self.dk).transpose(1, 2)
        V_h = V.view(B, L, self.h, self.dk).transpose(1, 2)
        
        scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(self.dk)
        
        token_to_chunk = torch.arange(L, device=device) // self.mem_len
        i_chunks = token_to_chunk.unsqueeze(1)
        j_chunks = token_to_chunk.unsqueeze(0)
        chunk_dist = (i_chunks - j_chunks).abs().float()
        
        r1 = F.softplus(self.r1_mem_self)
        r2 = F.softplus(self.r2_mem_self)
        r1 = torch.clamp(r1, max=10.0)
        r2 = torch.clamp(r2, max=10.0)
        
        kerple_bias = -r1.view(self.h, 1, 1) * torch.log1p(
            torch.clamp(r2.view(self.h, 1, 1) * chunk_dist.unsqueeze(0), max=1e4)
        )
        
        scores = scores + kerple_bias.unsqueeze(0)
        
        causal_mask = i_chunks < j_chunks
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.ones_like(attn) / attn.size(-1)
        attn = self.mem_self_dropout(attn)
        
        out = torch.matmul(attn, V_h)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d)
        
        out = self.mem_self_attn_o(out)
        out = self.mem_self_dropout(out)
        out = self.mem_self_ln1(mem_seq + out)
        
        ffn_out = self.mem_self_ffn(out)
        out = self.mem_self_ln2(out + ffn_out)
        
        return out 

    def forward(self, all_write_memories, Q_seq, L_padded):
        B = Q_seq.shape[0]
        device = Q_seq.device

        # 1. Memory Self-Attention
        KV_top = self.memory_self_attention(all_write_memories, device)
        if KV_top.size(1) == 0:
            return None # Fallback for empty memory
            
        # 2. Top Layer Cross-Attention
        Q_top = self.top_q_proj(Q_seq)
        K_top = self.top_k_proj(KV_top)
        V_top = self.top_v_proj(KV_top)

        L_top = Q_top.size(1)
        M_top = K_top.size(1)

        Q_h = Q_top.view(B, L_top, self.h, self.dk).permute(0, 2, 1, 3)
        K_h = K_top.view(B, M_top, self.h, self.dk).permute(0, 2, 1, 3)
        V_h = V_top.view(B, M_top, self.h, self.dk).permute(0, 2, 1, 3)

        logits_top = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(self.dk)

        q_chunk_indices = torch.arange(L_padded, device=device) // self.chunk_len
        k_token_indices = torch.arange(M_top, device=device)
        k_chunk_indices = k_token_indices // self.mem_len
        
        chunk_dist = (q_chunk_indices.unsqueeze(1) - k_chunk_indices.unsqueeze(0)).abs().float()
        r1 = F.softplus(self.r1_top)
        r2 = F.softplus(self.r2_top)
        r1 = torch.clamp(r1, max=10.0)
        r2 = torch.clamp(r2, max=10.0)
        
        kerple_bias = -r1.view(self.h, 1, 1) * torch.log1p(
            torch.clamp(r2.view(self.h, 1, 1) * chunk_dist.unsqueeze(0), max=1e4)
        )
        logits_top = logits_top + kerple_bias.unsqueeze(0)

        # Causal Retrieval Mask
        mask_cond = q_chunk_indices.unsqueeze(1) <= k_chunk_indices.unsqueeze(0)
        retrieval_mask = torch.zeros((L_padded, M_top), device=device)
        retrieval_mask = retrieval_mask.masked_fill(mask_cond, float('-inf'))
        logits_top = logits_top + retrieval_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(logits_top, dim=-1)
        if torch.isnan(attn_weights).any():
            attn_weights = torch.ones_like(attn_weights) / attn_weights.size(-1)
        attn_weights = self.top_dropout(attn_weights)

        top_output = torch.matmul(attn_weights, V_h)
        top_output = top_output.permute(0, 2, 1, 3).contiguous().view(B, L_top, self.d)
        
        top_attn_out = self.top_o_proj(top_output)
        top_attn_out = self.top_dropout(top_attn_out)
        top_output = self.top_ln1(Q_seq + top_attn_out)
        
        top_ffn_out = self.top_ffn(top_output)
        top_output = self.top_ln2(top_output + top_ffn_out)

        return top_output


class HRKT_FusionLayer(nn.Module):
    """
    Module for fusing Bottom (Local) and Top (Global) Representations
    """
    def __init__(self, d, dropout, top_dropout):
        super().__init__()
        self.top_dropout = nn.Dropout(top_dropout)
        self.fusion_ln = nn.LayerNorm(d)
        self.fusion_gate = nn.Linear(d * 2, d)
        
        self.final_ffn = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )

    def forward(self, bottom_output, top_output=None):
        if top_output is None:
            # Fallback when Top Layer is absent (e.g., Memory Size is 0)
            fused_features = self.fusion_ln(bottom_output)
        else:
            concat_features = torch.cat([bottom_output, top_output], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(concat_features))
            fused_features = bottom_output + self.top_dropout(gate * top_output)
            fused_features = self.fusion_ln(fused_features)

        logits = self.final_ffn(fused_features).squeeze(-1)
        output = torch.sigmoid(logits)
        return output

# ================================================================================================
# 3. Main Model: HRKT_SAKT (Final Wrapper Module)
# ================================================================================================

class HRKT_SAKT(nn.Module):
    def __init__(self, num_q, max_len, d, num_heads, dropout, chunk_len=50, mem_len=4, 
                 top_dropout=0.3, top_d_ff_ratio=4):
        super().__init__()
        self.num_q = num_q
        self.chunk_len = chunk_len

        # [Embeddings]
        self.E = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.M = nn.Embedding(2 * num_q + 1, d, padding_idx=0)

        # [HRKT Sub-Modules]
        self.bottom_layer = HRKT_BottomLayer(d, num_heads, dropout, chunk_len, mem_len)
        self.top_layer = HRKT_TopLayer(d, num_heads, dropout, top_dropout, mem_len, chunk_len, top_d_ff_ratio)
        self.fusion_layer = HRKT_FusionLayer(d, dropout, top_dropout)

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
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, q, r, t, qry):
        B, L = q.shape
        device = q.device

        # 1. Padding & Chunking (Data preprocessing logic controlled within the model)
        remainder = L % self.chunk_len
        if remainder != 0:
            pad_len = self.chunk_len - remainder
            pad_q = torch.zeros(B, pad_len, dtype=torch.long, device=device)
            pad_r = torch.zeros(B, pad_len, dtype=torch.long, device=device)
            pad_qry = torch.zeros(B, pad_len, dtype=torch.long, device=device)
            q = torch.cat([q, pad_q], dim=1)
            r = torch.cat([r, pad_r], dim=1)
            qry = torch.cat([qry, pad_qry], dim=1)

        L_padded = q.shape[1]
        num_chunks = L_padded // self.chunk_len
        
        if num_chunks == 0:
            return torch.zeros(B, L, device=device)
        
        q_chunks = q.reshape(B, num_chunks, self.chunk_len)
        r_chunks = r.reshape(B, num_chunks, self.chunk_len)
        qry_chunks = qry.reshape(B, num_chunks, self.chunk_len)

        # 2. Bottom Layer (Local memory processing)
        bottom_output, all_write_memories = self.bottom_layer(
            q_chunks, r_chunks, qry_chunks, self.E, self.M, self.num_q
        )

        # 3. Top Layer (Global memory processing)
        Q_seq = self.E(qry)
        if Q_seq.size(1) < L_padded:
            pad = L_padded - Q_seq.size(1)
            Q_seq = torch.cat([Q_seq, torch.zeros(B, pad, self.bottom_layer.d, device=device)], dim=1)

        top_output = self.top_layer(all_write_memories, Q_seq, L_padded)

        # 4. Fusion Layer (Merging and final prediction)
        output = self.fusion_layer(bottom_output, top_output)

        # Truncate the padded parts and return only the original sequence length
        return output[:, :L]

# ================================================================================================
# 4. Dummy Data Test (Functionality verification)
# ================================================================================================
if __name__ == "__main__":
    def random_data(bs, seq_len, total_q):
        q = torch.randint(1, total_q + 1, (bs, seq_len))
        r = torch.randint(0, 2, (bs, seq_len))
        t = torch.zeros(bs, seq_len) # dummy timestamp
        qry = torch.randint(1, total_q + 1, (bs, seq_len))
        return q, r, t, qry

    bs, seq_len, total_q = 64, 100, 1200
    q, r, t, qry = random_data(bs, seq_len, total_q)

    # Initialize Model (Base SAKT)
    sakt_model = SAKT(num_q=total_q, max_len=1000, d=128, num_heads=8, dropout=0.2)
    sakt_out = sakt_model(q, r, t, qry)
    print(f"Base SAKT Output Shape: {sakt_out.shape}")

    # Initialize Model (HRKT_SAKT)
    hrkt_model = HRKT_SAKT(
        num_q=total_q, max_len=1000, d=128, num_heads=8, dropout=0.2,
        chunk_len=50, mem_len=4
    )
    hrkt_out = hrkt_model(q, r, t, qry)
    print(f"HRKT SAKT Output Shape: {hrkt_out.shape}")
