import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#central = central vector, avg = average pooling, max = max pooling, average_attention = attention pooling, central_attention = central weighting
#other heads were not part of the grid search, but they are implemented here

class MLP(nn.Module):
    """Pooling + standard 2-layer MLP (shared across all heads)."""
    def __init__(self, input_dim, num_classes=5, mid_dim=None, dropout=0.1):
        super().__init__()
        mid_dim = max(128, input_dim // 2) if mid_dim is None else mid_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mid_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_classes)
        )

    def forward(self, pooled):
        return self.mlp(pooled)


class RNNClassifier(nn.Module):
    
    def __init__(
        self,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_classes=5,
        rnn_type="LSTM",
        bidirectional=True,
        dropout=0.1,
        pooling="avg",
        seq_len=51,
        kmer=1,
        embed="one_hot",

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.dropout_p = dropout
        self.pooling = pooling
        self.kmer = kmer

        self.seq_len = seq_len - kmer + 1

        if embed == "one_hot":
            self.input_channels = 4**kmer
            self.input_proj = nn.Linear(self.input_channels, embed_dim, bias=False)
        elif embed == "ENAC":
            self.input_channels = 4
            self.input_proj = nn.Linear(self.input_channels, embed_dim, bias=False)
        else:
            raise ValueError(f"Wrong embedding type for RNN: {embed}. It should be one_hot or ENAC.")


        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn_layers = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(num_layers):
            self.rnn_layers.append(
                rnn_cls(
                    input_size   = in_dim,
                    hidden_size  = hidden_dim,
                    num_layers   = 1,
                    batch_first  = True,
                    bidirectional= bidirectional,
                    dropout      = 0 #we do one layer at a time for masking, so we have to customize dropout
                )
            )
            in_dim = hidden_dim * (2 if bidirectional else 1)

        self.final_rnn_dim = in_dim

        self.inter_layer_drop = nn.Dropout(p=dropout)

        if self.pooling == "attention":
            self.att_proj = nn.Linear(self.final_rnn_dim, 1)

        if self.pooling == "central_attention":
            self.att_proj = nn.Linear(self.final_rnn_dim, self.seq_len)

        if self.pooling == "average_attention":
            self.to_q = nn.Linear(self.final_rnn_dim, self.final_rnn_dim, bias=False)
            self.to_k = nn.Linear(self.final_rnn_dim, self.final_rnn_dim, bias=False)
            self.to_v = nn.Linear(self.final_rnn_dim, self.final_rnn_dim, bias=False)

        if self.pooling == "average_multihead_attention":
            from .Transformer_rope import RotaryPositionEmbedding #ADD DOT LATER
            self.num_heads = 8
            self.c_head = self.final_rnn_dim // self.num_heads
            self.rotary_emb = RotaryPositionEmbedding(self.c_head)
            self.to_q = nn.Linear(self.final_rnn_dim, self.c_head * self.num_heads)
            self.to_k = nn.Linear(self.final_rnn_dim, self.c_head * self.num_heads)
            self.to_v = nn.Linear(self.final_rnn_dim, self.c_head * self.num_heads)
            self.output_proj = nn.Linear(self.c_head * self.num_heads, self.final_rnn_dim)

        if self.pooling == "self_attention":
            self.h_heads = 20
            self.ws1 = nn.Linear(self.final_rnn_dim, 350, bias=False)
            self.ws2 = nn.Linear(350, self.h_heads, bias=False)

        if self.pooling != "self_attention":
            self.mlp = MLP(input_dim=self.final_rnn_dim, num_classes=self.num_classes)
        else:
            self.mlp = MLP(input_dim=self.final_rnn_dim*self.h_heads, num_classes=self.num_classes, mid_dim=None, dropout=0.1)

    def forward(self, x, return_central_weights=False):

        if return_central_weights:
            assert self.pooling == "central_attention", "Central weights are only available for central weighted pooling."

        valid_mask = (x.abs().sum(-1) != 0) #shape: (batch_size, seq_len)

        x = self.input_proj(x)

        for i, rnn in enumerate(self.rnn_layers):
            x, _ = rnn(x)
            x    = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            if i < len(self.rnn_layers) - 1:   # skip last layer
                x = self.inter_layer_drop(x)   # custom dropout here


        if self.pooling == "central":
            central_idx = self.seq_len // 2
            center_output = x[:, central_idx, :]

            pooled = center_output

        elif self.pooling == "max":
            x = x.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
            pooled, _ = torch.max(x, dim=1)

        elif self.pooling == "avg":
            x = x*valid_mask.unsqueeze(-1)
            pooled = x.sum(1) / valid_mask.sum(1, keepdim=True)

        elif self.pooling == "attention":
            # Simple learned attention. For each position t, compute e_t = att_proj(rnn_out[:, t, :])
            att_scores = self.att_proj(x).squeeze(-1)
            att_scores = att_scores.masked_fill(~valid_mask, float("-inf"))
            att_weights = F.softmax(att_scores, dim=1)
            pooled = (x * att_weights.unsqueeze(2)).sum(dim=1)

        elif self.pooling == "central_attention":
            central_idx = self.seq_len // 2
            center_output = x[:, central_idx, :]
            central = self.att_proj(center_output) 
            central = F.gelu(central)
            central[:, central_idx] = float("-inf")
            central = central.masked_fill(~valid_mask, float("-inf"))
            attention_weights = torch.softmax(central, dim=1)
            pooled = (x * attention_weights.unsqueeze(2)).sum(dim=1)

        elif self.pooling == "average_attention":
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)

            attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            attn = attn.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
            attn = F.softmax(attn, dim=-1)

            z = torch.bmm(attn, v)
            z = z.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            pooled = z.sum(1) / valid_mask.sum(1, keepdim=True)

        elif self.pooling == "average_multihead_attention":
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
            output = self.output_proj(output)
            output = output*valid_mask.unsqueeze(-1)
            pooled = output.sum(1) / valid_mask.sum(1, keepdim=True)




        elif self.pooling == "self_attention":
            att_before = self.ws1(x)
            att_before = torch.tanh(att_before)
            att_before = att_before.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            att = self.ws2(att_before)
            att = att.masked_fill((~valid_mask).unsqueeze(-1), float("-inf"))
            att = F.softmax(att.transpose(1, 2), dim=-1)
            m = torch.bmm(att, x)
            #flatten the output
            pooled = m.view(m.shape[0], -1)



        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        logits = self.mlp(pooled)
        if return_central_weights:
            return logits, attention_weights
        else:
            return logits


if __name__ == "__main__":

    embed = "one_hot"
    kmer = 1
    seq_len=64
    if embed == "one_hot":
        input_dim = 4**kmer
    elif embed == "ENAC":
        input_dim = 4
    
    dummy_input = torch.randn(2, seq_len - kmer + 1, input_dim)
    #fake pad first token
    dummy_input[0, 0, :] = 0.0
    model = RNNClassifier(
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        rnn_type="GRU",
        bidirectional=True,
        dropout=0.1,
        pooling="average_attention",
        seq_len=seq_len,
        kmer=kmer,
        embed=embed,
    )
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)  # should be [2, 5]
