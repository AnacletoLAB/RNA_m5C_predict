import torch
import torch.nn as nn

# Code heavily inspired by https://blog.eleuther.ai/rotary-embeddings/, GPT-NeoX (Pytorch) implementation
# and ESM2 implementation https://github.com/facebookresearch/esm/blob/main/esm/rotary_embedding.py
#In this implementation, the x, y coordinate pairs for each 2D vector are: x_1 = seq[0], y_1 = seq[seq_len//2]; x_2 = seq[1], y_2 = seq[seq_len//2+1] etc.
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
    

# Example usage:
if __name__ == "__main__":
    # Example input
    batch_size = 1
    seq_len = 64
    dim = 16

    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)

    rotary_pos_emb = RotaryPositionEmbedding(dim)
    q_rotated, k_rotated = rotary_pos_emb(q, k)

    print("Rotated Q:", q_rotated)
    print("Rotated K:", k_rotated)
