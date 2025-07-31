import torch.nn as nn
import torch.nn.functional as F
import torch
import math

#central = central vector, avg = average pooling, max = max pooling, average_attention = attention pooling, central_attention = central weighting
#other heads were not part of the grid search, but they are implemented here

class MLP(nn.Module):
    """Pooling + standard 2-layer MLP (shared across all heads)."""
    def __init__(self, embed_dim, num_classes=5, mid_dim=None, dropout=0.1):
        super().__init__()
        mid_dim = max(128, embed_dim // 2) if mid_dim is None else mid_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_classes)
        )

    def forward(self, pooled):
        return self.mlp(pooled)

class ClassificationCentral(nn.Module):
    #central attention
    def __init__(self, embed_dim, seq_len=64, num_classes=5):
        super().__init__()

        self.seq_len = seq_len

        self.mlp = MLP(embed_dim, num_classes=num_classes)

    def forward(self, x, pad_mask=None):
        if pad_mask is None: #The model expects a pad_mask
            pad_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool).to(x.device)

        mid_point = self.seq_len // 2
        x = x[:, mid_point, :]
        return self.mlp(x)
    

class ClassificationAverage(nn.Module):
    #average pooling bigger MLP
    def __init__(self, embed_dim, seq_len=64, num_classes=5):
        super().__init__()

        self.seq_len = seq_len

        self.mlp = MLP(embed_dim, num_classes=num_classes)

    def forward(self, x, pad_mask=None):
        if pad_mask is None: #The model expects a pad_mask (for debugging purposes)
            pad_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool).to(x.device)

        valid_mask = ~pad_mask
        x = x*valid_mask.unsqueeze(-1)
        pooled = x.sum(1) / valid_mask.sum(1, keepdim=True)
        
        return self.mlp(pooled)


class ClassificationMax(nn.Module):
    #average + max pooling
    def __init__(self, embed_dim, seq_len=64, num_classes=5):
        super().__init__()

        self.seq_len = seq_len

        self.mlp = MLP(embed_dim, num_classes=num_classes)

    def forward(self, x, pad_mask=None):
        
        if pad_mask is None: #The model expects a pad_mask (for debugging purposes)
            pad_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool).to(x.device)


        x = x.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
        pooled, _ = torch.max(x, dim=1)

        return self.mlp(pooled)
    
class ClassificationAverageAttention(nn.Module):
        def __init__(self, embed_dim, seq_len=64, num_classes=5):
            super().__init__()

            self.seq_len = seq_len
            self.embed_dim = embed_dim
            self.to_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.to_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.to_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

            self.mlp = MLP(embed_dim, num_classes=num_classes)

        def forward(self, x, pad_mask=None, return_attn_weights=False):
            
            if pad_mask is None: #The model expects a pad_mask
                pad_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool).to(x.device)

            valid_mask = ~pad_mask
            x = x*valid_mask.unsqueeze(-1)

            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)

            attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            attn = attn.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
            attn = F.softmax(attn, dim=-1)

            z = torch.bmm(attn, v)
            z = z.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            pooled = z.sum(1) / valid_mask.sum(1, keepdim=True)
            pooled = self.mlp(pooled)
            
            if return_attn_weights:
                return pooled, attn
            else:
                return pooled


class ClassificationAttention(nn.Module):
    #with attention pooling
    def __init__(self, embed_dim, seq_len=64, num_classes=5):
        super().__init__()
        self.att_proj = nn.Linear(embed_dim, 1)

        self.seq_len = seq_len

        self.mlp = MLP(embed_dim, num_classes=num_classes)

    def forward(self, x, pad_mask=None):
        if pad_mask is None: #The model expects a pad_mask
            pad_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool).to(x.device)
            
        att_scores = self.att_proj(x).squeeze(-1)
        att_scores = att_scores.masked_fill(pad_mask, float("-inf"))
        att_weights = F.softmax(att_scores, dim=1)
        pooled = (x * att_weights.unsqueeze(2)).sum(dim=1)

        return self.mlp(pooled)
    

    
class ClassificationCNN(nn.Module):

    def __init__(self, embed_dim, seq_len=64, kmer=1, num_filters=[32, 64, 128], kernel_sizes=[3, 3, 3], pool_sizes=[1, 2, 2], do_dropout=True,
        drop_out_rate=0.2, num_classes=5):
        super().__init__()

        assert len(num_filters) == len(kernel_sizes) == len(pool_sizes), (
            "num_filters, kernel_sizes, and pool_sizes must have the same length!"
        )
        
        self.num_conv_layers = len(num_filters)
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.input_channels = embed_dim

        prev_channels = self.input_channels
        for nf, ks, ps in zip(num_filters, kernel_sizes, pool_sizes):
            # Conv1d args: in_channels, out_channels, kernel_size
            conv = nn.Conv1d(
                in_channels=prev_channels,
                out_channels=nf,
                kernel_size=ks,
                padding="same"
            )
            self.conv_layers.append(conv)
            # Pool layer
            pool = nn.MaxPool1d(kernel_size=ps, stride=ps)
            self.pool_layers.append(pool)

            prev_channels = nf  # next layer's in_channels = current out_channels

        self.seq_len = seq_len - kmer + 1
        dummy = torch.zeros(1, self.input_channels, self.seq_len)  # [B=1, C, L]
        with torch.no_grad():
            dummy_out = self._forward_features(dummy)
        out_shape = dummy_out.shape  # e.g. [1, final_channels, final_length]
        l_star = out_shape[1] * out_shape[2]

        self.dropout = nn.Dropout(drop_out_rate) if do_dropout else nn.Identity()

        self.mlp = MLP(embed_dim=l_star, num_classes=num_classes)

    def _forward_features(self, x):
        """
        forward pass to get the size of the output after conv/pool layers
        """
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        return x

    def forward(self, x, pad_mask=None):
        if pad_mask is None: #The model expects a pad_mask
            pad_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool).to(x.device)

        valid_mask = ~pad_mask
        x = x*valid_mask.unsqueeze(-1)

        x = x.transpose(1, 2)  # => [B, H, T]


        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = pool(x)  # reduces the seq_len by factor of pool_size


        x = x.reshape(x.size(0), -1)

        return  self.mlp(x)


    

class ClassificationCentralAttention(nn.Module):
    def __init__(self, embed_dim, seq_len=64, num_classes=5):
        super().__init__()

        self.seq_len = seq_len
        self.lin_att = nn.Linear(embed_dim, self.seq_len)
        self.mlp = MLP(embed_dim=embed_dim, num_classes=num_classes)


    def forward(self, x, pad_mask=None):
        if pad_mask is None: #The model expects a pad_mask
            pad_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool).to(x.device)
        
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        mid_point = self.seq_len // 2
        central = x[:, mid_point, :]
        central = self.lin_att(central)
        central = F.gelu(central)
        central[:, mid_point] = float("-inf")
        central = central.masked_fill(pad_mask, float("-inf"))
        att_weights = torch.softmax(central, dim=1)
        x = (x * att_weights.unsqueeze(2)).sum(dim=1)

        return self.mlp(x)
    
class ClassificationGRU(nn.Module):
    def __init__(self, embed_dim, seq_len, num_classes=5):
        super().__init__()

        self.seq_len = seq_len
        self.dropout_p = 0.1
        self.pooling = "central_attention"
        self.mlp_dim = 128

        self.num_layers = 1
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,    # input shape: [B, T, H]
            bidirectional=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0.0  # PyTorch applies dropout *between* stacked layers
        )

        embed_dim = embed_dim * 2 

        if self.pooling == "attention":
            self.att_proj = nn.Linear(embed_dim, 1)

        if self.pooling == "central_attention":
            self.att_proj = nn.Linear(embed_dim, self.seq_len)

        self.mlp = MLP(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x, pad_mask=None):
        if pad_mask is None: #The model expects a pad_mask
            pad_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool).to(x.device)

        # x: [B, T, H_d]
        x    = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x, _ = self.rnn(x)
        x    = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        if self.pooling == "attention":
            att_scores = self.att_proj(x).squeeze(-1)
            att_scores = att_scores.masked_fill(pad_mask, float("-inf"))
            att_weights = F.softmax(att_scores, dim=1)
            pooled = (x * att_weights.unsqueeze(2)).sum(dim=1)

        elif self.pooling == "central_attention":
            central_idx = self.seq_len // 2
            center_output = x[:, central_idx, :]
            central = self.att_proj(center_output) 
            central = F.gelu(central)
            central[:, central_idx] = float("-inf")
            central = central.masked_fill(pad_mask, float("-inf"))
            attention_weights = torch.softmax(central, dim=1)
            pooled = (x * attention_weights.unsqueeze(2)).sum(dim=1)

        elif self.pooling == "central":
            central_idx = self.seq_len // 2
            center_output = x[:, central_idx, :] #shape => [B, hidden_dim*2] or [B, hidden_dim]
            pooled = center_output

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")
        
        return self.mlp(pooled)



# A dictionary mapping head names to a lambda that returns a new head instance.
HEADS = {
    'central': lambda embed_dim, seq_len, num_classes: ClassificationCentral(embed_dim, seq_len, num_classes),
    'avg': lambda embed_dim, seq_len, num_classes: ClassificationAverage(embed_dim, seq_len, num_classes),
    'max': lambda embed_dim, seq_len, num_classes: ClassificationMax(embed_dim, seq_len, num_classes),
    'attention': lambda embed_dim, seq_len, num_classes: ClassificationAttention(embed_dim, seq_len, num_classes),
    '1DCNN': lambda embed_dim, seq_len, num_classes: ClassificationCNN(embed_dim, seq_len=seq_len, num_classes=num_classes),
    'central_attention': lambda embed_dim, seq_len, num_classes: ClassificationCentralAttention(embed_dim, seq_len, num_classes=num_classes),
    'GRU': lambda embed_dim, seq_len, num_classes: ClassificationGRU(embed_dim, seq_len, num_classes=num_classes),
    'average_attention': lambda embed_dim, seq_len, num_classes: ClassificationAverageAttention(embed_dim, seq_len, num_classes=num_classes),
}


if __name__ == "__main__":
    # Example usage
    #generate random tensor of shape (2, 64, 480)
    seq_len = 51
    x = torch.randn(2, seq_len, 480)
    #classification head
    head = ClassificationAverageAttention(480, seq_len)
    #forward pass
    output = head(x)
    print(output.shape)  # should be (2, 5)

