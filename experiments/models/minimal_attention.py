import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def dot_product_attention(q, k, v, attn_mask=None, key_pad_mask=None, dropout=None):
    c = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(c)

    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask, float("-inf"))

    if key_pad_mask is not None:
        attn = attn.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")) 

    attn = attn.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    output = torch.matmul(attn, v)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, c_in, num_heads, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()
        assert c_in % num_heads == 0, "Embedding dimensionality must be divisible with number of attention heads!"

        self.c_in = c_in
        self.num_heads = num_heads

        self.c_head = c_in // self.num_heads
        self.c_qkv = self.c_head * num_heads

        self.use_rot_emb = use_rot_emb
        if self.use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.c_head)

        self.to_q = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_k = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_v = nn.Linear(self.c_in, self.c_qkv, bias=bias)

        self.attention_dropout = nn.Dropout(p=attention_dropout)

        self.out_proj = nn.Linear(c_in, c_in, bias=bias)

    def forward(self, q, k, v, attn_mask=None, key_pad_mask=None):
        bs = q.shape[0]

        q = self.to_q(q).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        k = self.to_k(k).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        v = self.to_v(v).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)

        if self.use_rot_emb:
            q, k = self.rotary_emb(q, k)

        output, attn = dot_product_attention(q, k, v, attn_mask, key_pad_mask, self.attention_dropout)

        output = output.transpose(-2, -3).contiguous().view(bs, -1, self.num_heads * self.c_head)
        output = self.out_proj(output)

        return output, attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, c_in, num_heads, attention_dropout=0.0, use_rot_emb=True, bias=False):
        super().__init__()

        self.mh_attn = MultiHeadAttention(c_in, num_heads, attention_dropout, use_rot_emb, bias)

    def forward(self, x, attn_mask=None, key_pad_mask=None):
        return self.mh_attn(x, x, x, attn_mask, key_pad_mask)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1) # x: [B, T, 2*dim] => x1: [B, T, dim], x2: [B, T, dim]
    return torch.cat((-x2, x1), dim=-1) # returns [B, T, 2*dim] tensor

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: int = 10000
    ):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq) #tracks in state dict, but not in optimizer and automatically moved to GPU if needed

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cached(self, x, seq_dim):
        seq_len = x.shape[seq_dim]

        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) # [0, 1, 2, ..., seq_len-1]
            freqs = torch.einsum("i,j->ij", t, self.inv_freq) # [seq_len, dim/2]
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q, k):
        self._update_cached(k, seq_dim=-2)
        return apply_rotary_pos_emb(q, k, self.cos_cached, self.sin_cached)



class minimal_Attention(nn.Module):
    def __init__(self, seq_len, c_in, num_heads):
        super(minimal_Attention, self).__init__()
        self.seq_len = seq_len
        self.input_channels = 4
        self.c_in = c_in
        self.num_heads = num_heads

        self.c_head = c_in // self.num_heads
        self.c_qkv = self.c_head * num_heads

        self.input_proj = nn.Linear(self.input_channels, c_in, bias=False)

        self.to_q = nn.Linear(self.c_in, self.c_qkv)
        self.to_k = nn.Linear(self.c_in, self.c_qkv)
        self.to_v = nn.Linear(self.c_in, self.c_qkv)

        self.rotary_emb = RotaryPositionEmbedding(self.c_head)

        self.out_proj = nn.Linear(c_in, c_in)

        self.mlp = nn.Linear(c_in, c_in//2)
        self.mlp_2 = nn.Linear(c_in//2, 5)


    def forward(self, x):
        
        valid_mask = (x.abs().sum(-1) != 0) #shape: (batch_size, seq_len)
        x = self.input_proj(x)

        bs = x.shape[0]
        q = self.to_q(x).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        k = self.to_k(x).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)
        v = self.to_v(x).view(bs, -1, self.num_heads, self.c_head).transpose(-2, -3)

        q, k = self.rotary_emb(q, k)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        attn = attn.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = attn.softmax(dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(-2, -3).contiguous().view(bs, -1, self.num_heads * self.c_head)
        output = self.out_proj(output)
        mid_point = self.seq_len // 2
        pooled = output[:, mid_point, :]
        # output = output*valid_mask.unsqueeze(-1)
        # pooled = output.sum(1) / valid_mask.sum(1, keepdim=True)
        pooled = F.gelu(self.mlp(pooled))
        pooled = self.mlp_2(pooled)
        return pooled


if __name__ == "__main__":
    seq_len = 51
    batch_size = 2
    x_hidden = torch.randn(2, seq_len, 4)
    x_hidden[0, 0] = 0

    model = minimal_Attention(seq_len=seq_len, c_in=300, num_heads=10)
    logits = model(x_hidden)
    print("Logits shape:", logits.shape)  # should be [2, 5]