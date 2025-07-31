import torch
from torch import nn
from torch.nn import functional as F

from .Transformer_attention import MultiHeadSelfAttention, FlashMultiHeadSelfAttention


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, 
                 attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()

        self.use_flash_attn = use_flash_attn

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, use_rot_emb, attn_qkv_bias, transition_dropout, attention_dropout, 
                                 residual_dropout, transition_factor, use_flash_attn) for _ in range(num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, need_attn_weights=False):
        attn_weights = None
        if need_attn_weights:
            attn_weights = []

        for block in self.blocks:

            x, attn = block(x, key_padding_mask=key_padding_mask, 
                need_attn_weights=need_attn_weights)

            if need_attn_weights:
                attn_weights.append(attn)


        x = self.final_layer_norm(x)

        return x, attn_weights

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    In the cited paper beta is set to 1 and is not learnable;
    but by the Swish definition it is learnable parameter otherwise
    it is SiLU activation function (https://paperswithcode.com/method/swish)
    """
    def __init__(self, size_in, size_out, beta_is_learnable=True, bias=True):
        """
        Args:
            size_in: input embedding dimension
            size_out: output embedding dimension
            beta_is_learnable: whether beta is learnable or set to 1, learnable by default
            bias: whether use bias term, enabled by default
        """
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.linear_gate = nn.Linear(size_in, size_out, bias=bias)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=beta_is_learnable)  

    def forward(self, x):
        linear_out = self.linear(x)
        swish_out = linear_out * torch.sigmoid(self.beta * linear_out)
        return swish_out * self.linear_gate(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()
        
        self.use_flash_attn = use_flash_attn

        if use_flash_attn:
            self.mh_attn = FlashMultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, causal=False, use_rot_emb=use_rot_emb, bias=attn_qkv_bias)
        else:
            self.mh_attn = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout,  use_rot_emb, attn_qkv_bias)
        
        self.attn_layer_norm = nn.LayerNorm(embed_dim)

        self.transition = nn.Sequential(
                SwiGLU(embed_dim, int(2 / 3 * transition_factor * embed_dim), beta_is_learnable=True, bias=True),
                nn.Dropout(p=transition_dropout),
                nn.Linear(int(2 / 3 * transition_factor * embed_dim), embed_dim, bias=True),
        )
        self.out_layer_norm = nn.LayerNorm(embed_dim)

        self.residual_dropout_1 = nn.Dropout(p=residual_dropout)
        self.residual_dropout_2 = nn.Dropout(p=residual_dropout)

    def forward(self, x, key_padding_mask=None, need_attn_weights=None):
        x = self.attn_layer_norm(x)
        if self.use_flash_attn:
            mh_out, attn = self.mh_attn(x, key_padding_mask=key_padding_mask, return_attn_probs=need_attn_weights)
        else:
            mh_out, attn = self.mh_attn(x, attn_mask=None, key_pad_mask=key_padding_mask)
        x = x + self.residual_dropout_1(mh_out)

        residual = x
        x = self.out_layer_norm(x)
        x = residual + self.residual_dropout_2(self.transition(x))

        return x, attn