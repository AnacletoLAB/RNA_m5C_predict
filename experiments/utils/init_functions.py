import torch
import torch.nn as nn
import torch.nn.init as init


def init_func(module):
    if isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            # weight_ih*: input-hidden (x -> hidden)
            if "weight_ih" in name:
                # Xavier or Kaiming for the input→hidden matrix
                nn.init.xavier_uniform_(param.data)

            # weight_hh*: hidden-hidden (hidden[t-1] -> hidden[t])
            elif "weight_hh" in name:
                # Ortho is often used for recurrent kernels
                nn.init.orthogonal_(param.data)

            elif "bias" in name:
                nn.init.zeros_(param.data)

    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        embedding_dim = module.weight.size(1)
        bound = 1.0 / (embedding_dim ** 0.5)
        nn.init.uniform_(module.weight, -bound, bound)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def initialize_head(modules):
    """
    Traverse the new head's submodules and apply a custom initialization
    depending on the module type.
    """
    for name, module in modules.named_modules():
        if isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                # weight_ih*: input-hidden (x -> hidden)
                if "weight_ih" in name:
                    # Xavier or Kaiming for the input→hidden matrix
                    nn.init.xavier_uniform_(param.data)

                # weight_hh*: hidden-hidden (hidden[t-1] -> hidden[t])
                elif "weight_hh" in name:
                    # Ortho is often used for recurrent kernels
                    nn.init.orthogonal_(param.data)

                elif "bias" in name:
                    nn.init.zeros_(param.data)

        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            embedding_dim = module.weight.size(1)
            bound = 1.0 / (embedding_dim ** 0.5)
            nn.init.uniform_(module.weight, -bound, bound)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

