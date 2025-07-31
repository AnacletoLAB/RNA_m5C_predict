import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
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


class CNNClassifier(nn.Module):
    """
    1D CNN for classification of one-hot-encoded RNA sequences.
    """

    def __init__(
        self,
        num_filters=[32, 64],
        kernel_sizes=[5, 5],
        pool_sizes=[2, 2],
        seq_len=64,
        num_classes=5,
        do_dropout=True,
        drop_out_rate=0.2,
        kmer=1,
        embed="one_hot",
    ):
        super().__init__()

        assert len(num_filters) == len(kernel_sizes) == len(pool_sizes), (
            "num_filters, kernel_sizes, and pool_sizes must have the same length!"
        )

        self.num_conv_layers = len(num_filters)
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.kmer = kmer
        self.num_classes = num_classes
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
            # Pool layer
            pool = nn.MaxPool1d(kernel_size=ps, stride=ps)
            self.pool_layers.append(pool)

            prev_channels = nf  # next layer's in_channels = current out_channels

        
        self.seq_len = seq_len - kmer + 1

        dummy = torch.zeros(1, self.input_channels, self.seq_len)
        with torch.no_grad():
            dummy_out = self._forward_features(dummy)
        out_shape = dummy_out.shape
        l_star = out_shape[1] * out_shape[2]


        self.mlp = MLP(l_star, num_classes=self.num_classes, mid_dim=None, dropout=0.1)

        self.dropout = nn.Dropout(drop_out_rate) if do_dropout else nn.Identity()

    def _forward_features(self, x):
        """
        forward pass to get the size of the output after conv/pool layers
        """
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        return x

    def forward(self, x):

        x = x.permute(0, 2, 1)  # [B, input_channels, seq_len]

        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = pool(x)
            

        # x is now [B, final_filters, final_seq_len]
        # Flatten fully: [B, final_filters * final_seq_len]
        x = x.reshape(x.size(0), -1)


        x = self.mlp(x)

        return x


if __name__ == "__main__":
    # Dummy test
    seq_len = 64
    kmer = 1
    embed = "ENAC"
    if embed == "one_hot":
        input_dim = 4**kmer
    elif embed == "ENAC":
        input_dim = 4
    dummy_input = torch.randn(1, seq_len - kmer + 1, input_dim) 

    model = CNNClassifier(
        num_filters=[32, 64, 128],
        kernel_sizes=[4, 4, 4],
        pool_sizes=[1, 1, 1],
        num_classes=5,
        do_dropout=True,
        kmer=1,
        seq_len=seq_len,
        embed=embed,
    )
    logits = model(dummy_input)
    print("Logits shape:", logits.shape) 
