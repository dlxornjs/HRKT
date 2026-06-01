import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================================================
# 1. HRKT Plugin Core Components (Independent Module for Plug-and-Play)
# ================================================================================================

class KerpleBias(nn.Module):
    """Kerple bias for bottom layer (RMT chunks) and memory refinement"""
    def __init__(self, n_heads):
        super().__init__()
        self.tau = nn.Parameter(torch.randn(n_heads, 1, 1))
        self.sigma = nn.Parameter(torch.randn(n_heads, 1, 1))
    
    def forward(self, T, device):
        dist = torch.arange(T, device=device).unsqueeze(0) - torch.arange(T, device=device).unsqueeze(1)
        dist = torch.abs(dist).float()
        bias = -torch.exp(self.tau) * torch.log(1 + torch.exp(self.sigma) * dist)
        return bias

class KerpleTopBias(nn.Module):
    """Kerple bias for top layer (cross-attention with memory tokens)"""
    def __init__(self, n_heads):
        super().__init__()
        self.tau = nn.Parameter(torch.randn(n_heads, 1, 1))
        self.sigma = nn.Parameter(torch.randn(n_heads, 1, 1))
    
    def forward(self, L, M, device):
        dist = torch.arange(L, device=device).unsqueeze(1) - torch.arange(M, device=device).unsqueeze(0)
        dist = torch.abs(dist).float()
        bias = -torch.exp(self.tau) * torch.log(1 + torch.exp(self.sigma) * dist)
        return bias


class HRKT_BottomLayer(nn.Module):
    """Handles local chunk-level information and Recurrent Memory (RMT) propagation"""
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, chunk_len, mem_len):
        super().__init__()
        self.d, self.h, self.dk = d_model, n_heads, d_feature
        self.chunk_len, self.mem_len = chunk_len, mem_len
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.attn_dropout = nn.Dropout(dropout)
        self.kerple = KerpleBias(n_heads)

    def _create_causal_mask(self, T, mask_type, device):
        M, C = self.mem_len, self.chunk_len
        diagonal = 0 if mask_type == 0 else 1
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=diagonal)
        causal[:M, :] = False
        causal[M+C:, :] = False
        return causal

    def forward(self, seq_q, seq_k, seq_v, mask_type, device):
        B, T, D = seq_q.shape
        
        Q = self.q_proj(seq_q).view(B, T, self.h, self.dk).transpose(1, 2)
        K = self.k_proj(seq_k).view(B, T, self.h, self.dk).transpose(1, 2)
        V = self.v_proj(seq_v).view(B, T, self.h, self.dk).transpose(1, 2)
        
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        logits = logits + self.kerple(T, device).unsqueeze(0)
        
        causal_mask = self._create_causal_mask(T, mask_type, device)
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(logits, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.ones_like(attn) / max(attn.size(-1), 1)
            
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, V).transpose(1, 2).reshape(B, T, D)
        attn_out = self.o_proj(attn_out)
        
        out1 = self.ln1(seq_q + attn_out)
        out2 = self.ln2(out1 + self.ffn(out1))
        
        return out2


class HRKT_TopLayer(nn.Module):
    """Handles global context via chunk-level summary memory self-attention & cross-attention"""
    def __init__(self, d_model, d_feature, n_heads, dropout, top_dropout, top_d_ff_ratio, chunk_len, mem_len):
        super().__init__()
        self.d, self.h, self.dk = d_model, n_heads, d_feature
        self.chunk_len, self.mem_len = chunk_len, mem_len
        
        # Memory Refinement
        self.refine_q = nn.Linear(d_model, d_model)
        self.refine_k = nn.Linear(d_model, d_model)
        self.refine_v = nn.Linear(d_model, d_model)
        self.refine_o = nn.Linear(d_model, d_model)
        self.refine_ln1 = nn.LayerNorm(d_model)
        self.refine_ln2 = nn.LayerNorm(d_model)
        
        mem_d_ff = int(d_model * 2)
        self.refine_ffn = nn.Sequential(
            nn.Linear(d_model, mem_d_ff), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(mem_d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.refine_attn_drop = nn.Dropout(dropout)
        self.refine_bias = KerpleBias(n_heads)
        
        # Top Cross-Attention
        self.top_q = nn.Linear(d_model, d_model)
        self.top_k = nn.Linear(d_model, d_model)
        self.top_v = nn.Linear(d_model, d_model)
        self.top_o = nn.Linear(d_model, d_model)
        self.top_ln1 = nn.LayerNorm(d_model)
        self.top_ln2 = nn.LayerNorm(d_model)
        
        top_d_ff = int(d_model * top_d_ff_ratio)
        self.top_ffn = nn.Sequential(
            nn.Linear(d_model, top_d_ff), 
            nn.ReLU(),
            nn.Dropout(top_dropout), 
            nn.Linear(top_d_ff, d_model),
            nn.Dropout(top_dropout)
        )
        self.top_attn_drop = nn.Dropout(top_dropout)
        self.kerple_top = KerpleTopBias(n_heads)

    def refine_memories(self, memories_list, device):
        N = len(memories_list)
        if N == 0:
            return torch.zeros(1, 0, self.d, device=device)
        
        stacked = torch.stack(memories_list, dim=1)
        B, num_chunks, M, D = stacked.shape
        memories = stacked.view(B, num_chunks * M, D)
        T = memories.size(1)
        
        Q = self.refine_q(memories).view(B, T, self.h, self.dk).transpose(1, 2)
        K = self.refine_k(memories).view(B, T, self.h, self.dk).transpose(1, 2)
        V = self.refine_v(memories).view(B, T, self.h, self.dk).transpose(1, 2)
        
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        logits = logits + self.refine_bias(T, device).unsqueeze(0)
        
        chunk_indices = torch.arange(T, device=device) // M
        causal_mask = chunk_indices.unsqueeze(1) < chunk_indices.unsqueeze(0)
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(logits, dim=-1)
        if torch.isnan(attn).any():
            attn = torch.ones_like(attn) / max(attn.size(-1), 1)
            
        attn_out = torch.matmul(self.refine_attn_drop(attn), V).transpose(1, 2).reshape(B, T, D)
        attn_out = self.refine_o(attn_out)
        
        out1 = self.refine_ln1(memories + attn_out)
        out2 = self.refine_ln2(out1 + self.refine_ffn(out1))
        return out2

    def forward(self, query, all_write_memories, device):
        refined_mem = self.refine_memories(all_write_memories, device)
        M_total = refined_mem.size(1)
        
        if M_total == 0:
            return query, refined_mem
        
        B, L, D = query.shape
        Q = self.top_q(query).view(B, L, self.h, self.dk).transpose(1, 2)
        K = self.top_k(refined_mem).view(B, M_total, self.h, self.dk).transpose(1, 2)
        V = self.top_v(refined_mem).view(B, M_total, self.h, self.dk).transpose(1, 2)
        
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        logits = logits + self.kerple_top(L, M_total, device).unsqueeze(0)
        logits = torch.clamp(logits, min=-1e4, max=1e4)
        
        q_chunk_idx = torch.arange(L, device=device) // self.chunk_len
        k_chunk_idx = torch.arange(M_total, device=device) // self.mem_len
        causal_mask = q_chunk_idx.unsqueeze(1) < k_chunk_idx.unsqueeze(0)
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(logits, dim=-1)
        if torch.isnan(attn).any() or torch.isinf(attn).any():
            attn = torch.ones_like(attn) / max(attn.size(-1), 1)
            
        attn_out = torch.matmul(self.top_attn_drop(attn), V).transpose(1, 2).reshape(B, L, D)
        attn_out = self.top_o(attn_out)
        
        out1 = self.top_ln1(query + attn_out)
        out2 = self.top_ln2(out1 + self.top_ffn(out1))
        return out2, refined_mem


class HRKT_FusionLayer(nn.Module):
    """Fuses Local (Bottom) and Global (Top) representations dynamically"""
    def __init__(self, d_model, top_dropout):
        super().__init__()
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        self.fusion_ln = nn.LayerNorm(d_model)
        self.fusion_dropout = nn.Dropout(top_dropout)

    def forward(self, bottom_out, top_out):
        if top_out is None:
            return self.fusion_ln(bottom_out)
            
        concat = torch.cat([bottom_out, top_out], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(concat))
        fused = bottom_out + self.fusion_dropout(gate * top_out)
        return self.fusion_ln(fused)

# ================================================================================================
# 2. The Plugin Wrapper (HRKT_Plugin)
# ================================================================================================
class HRKT_Plugin(nn.Module):
    """
    Plug-and-play module that replaces standard Self-Attention in Transformer-based KT models.
    """
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, 
                 chunk_len=50, mem_len=4, top_dropout=0.3, top_d_ff_ratio=4):
        super().__init__()
        self.chunk_len = chunk_len
        self.mem_len = mem_len
        self.d_model = d_model
        
        self.read_mem_init = nn.Parameter(torch.randn(1, mem_len, d_model) * 0.02)
        self.write_mem_token = nn.Parameter(torch.randn(1, mem_len, d_model) * 0.02)
        
        self.bottom = HRKT_BottomLayer(d_model, d_feature, d_ff, n_heads, dropout, chunk_len, mem_len)
        self.top = HRKT_TopLayer(d_model, d_feature, n_heads, dropout, top_dropout, top_d_ff_ratio, chunk_len, mem_len)
        self.fusion = HRKT_FusionLayer(d_model, top_dropout)
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, mask, query, key, values, apply_pos=True):
        B, L, D = query.shape
        device = query.device
        
        # 1. Padding to fit chunk_len
        remainder = L % self.chunk_len
        pad_len = (self.chunk_len - remainder) if remainder != 0 else 0
        if pad_len > 0:
            query = torch.cat([query, torch.zeros(B, pad_len, D, device=device)], dim=1)
            key = torch.cat([key, torch.zeros(B, pad_len, D, device=device)], dim=1)
            values = torch.cat([values, torch.zeros(B, pad_len, D, device=device)], dim=1)
        
        num_chunks = query.shape[1] // self.chunk_len
        if num_chunks == 0:
            return self.fusion(query, None)[:, :L, :]
        
        # 2. Chunking
        q_chunks = query.reshape(B, num_chunks, self.chunk_len, D)
        k_chunks = key.reshape(B, num_chunks, self.chunk_len, D)
        v_chunks = values.reshape(B, num_chunks, self.chunk_len, D)
        
        current_read_mem = self.read_mem_init.expand(B, -1, -1)
        bottom_outputs_data = []
        all_write_memories = []
        
        # 3. Bottom Layer Sequential Processing
        for i in range(num_chunks):
            q_c, k_c, v_c = q_chunks[:, i, :], k_chunks[:, i, :], v_chunks[:, i, :]
            
            full_seq_q = torch.cat([current_read_mem, q_c, self.write_mem_token.expand(B, -1, -1)], dim=1)
            full_seq_k = torch.cat([current_read_mem, k_c, self.write_mem_token.expand(B, -1, -1)], dim=1)
            full_seq_v = torch.cat([current_read_mem, v_c, self.write_mem_token.expand(B, -1, -1)], dim=1)
            
            out = self.bottom(full_seq_q, full_seq_k, full_seq_v, mask, device)
            
            bottom_outputs_data.append(out[:, self.mem_len:-self.mem_len, :])
            current_read_mem = out[:, -self.mem_len:, :]
            all_write_memories.append(current_read_mem)
            
        bottom_output = torch.cat(bottom_outputs_data, dim=1)
        
        # 4. Top Layer (Refinement & Cross-Attention)
        top_output, refined_mem = self.top(query, all_write_memories, device)
        
        # 5. Fusion Layer
        if refined_mem.size(1) == 0:
            top_output = None
            
        fused = self.fusion(bottom_output, top_output)
        
        # 6. Truncate padding and return
        return fused[:, :L, :]

# ================================================================================================
# 3. Application Examples (Base SAKT and Assembled HRKT_SAKT)
# ================================================================================================

class SAKT_Base(nn.Module):
    """Pure Original SAKT without Kerple (for accurate baseline comparisons)"""
    def __init__(self, num_q, max_len, d, num_heads, dropout):
        super().__init__()
        self.num_q = num_q
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

    def forward(self, q, r, t, qry):
        B, L = q.shape
        x = q + self.num_q * r
        M = self.M(x)
        E = self.E(qry)

        pos = self.P[:L].unsqueeze(0)
        E = E + pos
        M = M + pos

        Q = self.W_Q(E).view(B, L, self.h, self.dk).permute(0, 2, 1, 3)
        K = self.W_K(M).view(B, L, self.h, self.dk).permute(0, 2, 1, 3)
        V = self.W_V(M).view(B, L, self.h, self.dk).permute(0, 2, 1, 3)

        raw = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        causal = torch.triu(torch.ones(L, L, device=raw.device, dtype=torch.bool), diagonal=1)
        raw = raw.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(raw, dim=-1)
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, L, self.d)
        out = self.W_O(out)

        out = self.ln1(out + E)
        out = self.ln2(out + self.ffn(out))

        logits = self.pred(out).squeeze(-1)
        return torch.sigmoid(logits)


class HRKT_SAKT(nn.Module):
    """SAKT integrated with HRKT Plugin Framework"""
    def __init__(self, num_q, max_len, d, num_heads, dropout, chunk_len=50, mem_len=4):
        super().__init__()
        self.num_q = num_q
        self.d = d
        
        self.E = nn.Embedding(num_q + 1, d, padding_idx=0)
        self.M = nn.Embedding(2 * num_q + 1, d, padding_idx=0)
        self.P = nn.Parameter(torch.Tensor(max_len + 1, d))
        nn.init.kaiming_normal_(self.P)
        
        self.pred = nn.Linear(d, 1)
        
        # Injecting the HRKT Plugin
        self.hrkt_layer = HRKT_Plugin(
            d_model=d, d_feature=d // num_heads, d_ff=d, n_heads=num_heads, 
            dropout=dropout, kq_same=False, chunk_len=chunk_len, mem_len=mem_len
        )

    def forward(self, q, r, t, qry):
        B, L = q.shape
        x = q + self.num_q * r
        
        M_emb = self.M(x)
        E_emb = self.E(qry)
        
        pos = self.P[:L].unsqueeze(0)
        E_emb = E_emb + pos
        M_emb = M_emb + pos
        
        # Execute HRKT Plugin (mask=1 indicates standard causal masking)
        fused_out = self.hrkt_layer(mask=1, query=E_emb, key=M_emb, values=M_emb)
        
        logits = self.pred(fused_out).squeeze(-1)
        return torch.sigmoid(logits)

# ================================================================================================
# 4. Dummy Testing Block
# ================================================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs, seq_len, total_q = 8, 120, 1000
    
    q = torch.randint(1, total_q + 1, (bs, seq_len)).to(device)
    r = torch.randint(0, 2, (bs, seq_len)).to(device)
    t = torch.zeros(bs, seq_len).to(device)
    qry = torch.randint(1, total_q + 1, (bs, seq_len)).to(device)
    
    sakt_baseline = SAKT_Base(num_q=total_q, max_len=1000, d=64, num_heads=4, dropout=0.2).to(device)
    base_out = sakt_baseline(q, r, t, qry)
    print(f"SAKT_Base Output Shape: {base_out.shape}")
    
    hrkt_sakt = HRKT_SAKT(num_q=total_q, max_len=1000, d=64, num_heads=4, dropout=0.2, chunk_len=50, mem_len=4).to(device)
    hrkt_out = hrkt_sakt(q, r, t, qry)
    print(f"HRKT_SAKT Output Shape: {hrkt_out.shape}")
