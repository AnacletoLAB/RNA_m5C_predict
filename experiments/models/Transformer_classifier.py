
import torch
from torch import nn
import torch.nn.functional as F
from .Transformer_modules import Transformer
from .Transformer_heads import HEADS

class Transformer_classifier(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, head_type="average_attention", seq_len=64, kmer=1, embed="one_hot", num_classes=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.use_flash_attn = True
        self.kmer = kmer
        self.embed = embed
        self.seq_len = seq_len - kmer + 1
        self.head_type = head_type
        self.num_classes = num_classes

        if embed == "one_hot":
            self.input_channels = 4**kmer
            self.input_proj = nn.Linear(self.input_channels, embed_dim, bias=False)
        elif embed == "ENAC":
            self.input_channels = 4
            self.input_proj = nn.Linear(self.input_channels, embed_dim, bias=False)
        elif embed == "embeddings":
            self.input_channels = 22
            self.pad_tkn_idx = 0
            self.embedding = nn.Embedding(self.input_channels, embed_dim, padding_idx=self.pad_tkn_idx)


        self.transformer = Transformer(attn_qkv_bias=False, num_blocks=self.num_blocks, num_heads=self.num_heads, embed_dim=self.embed_dim, use_rot_emb=True,
                                        transition_dropout=0.0, attention_dropout=0.1, residual_dropout=0.1, use_flash_attn=self.use_flash_attn)
        
        

        self.head = HEADS[head_type](self.embed_dim, seq_len=self.seq_len, num_classes=self.num_classes)

    def forward(self, tokens, need_attn_weights=False, final_attn_weights=False):

        if self.embed != "embeddings":
            pad_mask = ~(tokens.abs().sum(-1) != 0)
            x = self.input_proj(tokens)
        else:
            pad_mask = (tokens == self.pad_tkn_idx)
            x = self.embedding(tokens)

        if self.use_flash_attn:
            representation, attn_weights = self.transformer(
                x,
                key_padding_mask=torch.logical_not(pad_mask) if pad_mask is not None else None,
                need_attn_weights=need_attn_weights
            )
        else:
            representation, attn_weights = self.transformer(
                x,
                key_padding_mask=pad_mask,
                need_attn_weights=need_attn_weights
            )

        if final_attn_weights:
            assert self.head_type == "average_attention", "Final attention weights are only available for average_attention head type."

            logits, attn_weights = self.head(representation, pad_mask=pad_mask, return_attn_weights=True)
            return logits, attn_weights

        else:
            logits = self.head(representation, pad_mask=pad_mask)
            return logits
    

#example usage
if __name__ == "__main__":
    seq_len=64 
    kmer=2
    embed="ENAC"
    model = Transformer_classifier(embed_dim=120, num_blocks=2, num_heads=5, head_type="classificationCentral", seq_len=seq_len, kmer=kmer, embed=embed)
    model = model.to("cuda")
    seq_len = seq_len - kmer + 1
    input_channels = 4**kmer if embed == "one_hot" else 4
    # Dummy input
    x = torch.randn(32, seq_len, input_channels)  # Example input (batch_size=32, seq_len=seq_len, input_channels=input_channels)
    if embed == "embeddings":
        x = x.long()
        x = torch.randint(0, 4, (32, seq_len))
    with torch.cuda.amp.autocast():
        logits = model(x.to("cuda"))
    print(logits.shape)
