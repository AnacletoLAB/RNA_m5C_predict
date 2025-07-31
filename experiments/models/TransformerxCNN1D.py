
import torch
from torch import nn
import torch.nn.functional as F
from .Transformer_modules import Transformer
from .Transformer_heads import HEADS

class TransformerxCNN1D(nn.Module):
    def __init__(self, num_blocks, num_heads, head_type="classification", seq_len=64, kmer=1, embed="one_hot",
                num_filters=[32, 64], kernel_sizes=[5, 5], pool_sizes=[2, 2], do_dropout=True, drop_out_rate=0.2,):
        
        super().__init__()
        self.kmer=kmer
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.use_flash_attn = True
        self.kmer = kmer

        self.seq_len = seq_len - self.kmer + 1


        self.num_conv_layers = len(num_filters)
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        if embed == "one_hot":
            self.input_channels = 4**self.kmer
        elif embed == "ENAC":
            self.input_channels = 4

        prev_channels = self.input_channels
        for nf, ks, ps in zip(num_filters, kernel_sizes, pool_sizes):
            conv = nn.Conv1d(
                in_channels=prev_channels,
                out_channels=nf,
                kernel_size=ks,
                padding="same"
            )
            self.conv_layers.append(conv)
            pool = nn.MaxPool1d(kernel_size=ps, stride=ps)
            self.pool_layers.append(pool)

            prev_channels = nf  # next layer's in_channels = current out_channels

        
        dummy = torch.zeros(1, self.input_channels, seq_len - kmer + 1)  # [B=1, C, L]
        with torch.no_grad():
            dummy_out = self._forward_features(dummy)
        out_shape = dummy_out.shape  # e.g. [1, final_channels, final_length]
        self.seq_len = out_shape[2]
        self.embed_dim = out_shape[1]

        self.dropout = nn.Dropout(drop_out_rate) if do_dropout else nn.Identity()


        self.transformer = Transformer(attn_qkv_bias=False, num_blocks=self.num_blocks, num_heads=self.num_heads, embed_dim=self.embed_dim, use_rot_emb=True,
                                        transition_dropout=0.0, attention_dropout=0.1, residual_dropout=0.1, use_flash_attn=self.use_flash_attn)
        
        
        # Decide which head to attach.
        self.head = HEADS[head_type](self.embed_dim, seq_len=self.seq_len)

    def _forward_features(self, x):
        """
        forward pass to get the size of the output after conv/pool layers
        """
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        return x

    def forward(self, tokens, need_attn_weights=False):
        
        pad_mask = None

        x = tokens.permute(0, 2, 1)  # [B, input_channels, seq_len]

        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = pool(x) 

        x = x.permute(0, 2, 1)  # [B, seq_len, input_channels]
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

        # Pass transformer representation through the head
        logits = self.head(representation, pad_mask=pad_mask)
        # result = {"logits": logits, "representation": representation}
        # if need_attn_weights:
        #     result["attentions"] = torch.stack(attn_weights, dim=1)
            
        return logits
    

#example usage
if __name__ == "__main__":
    seq_len=64 
    kmer=3
    embed="one_hot"
    model = TransformerxCNN1D(num_blocks=2, num_heads=8, head_type="classification6", seq_len=seq_len, kmer=kmer, embed=embed
                              , num_filters=[32, 64], kernel_sizes=[5, 5], pool_sizes=[1, 1])
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
