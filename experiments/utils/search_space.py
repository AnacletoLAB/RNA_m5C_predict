from sklearn.model_selection import ParameterGrid
import itertools

import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

kmer = [1]
embed = ["one_hot"]
seq_len = [101]

global_search_space = {
    "RNN": {
        "batch_size": [16,32],
        "base_seed": [42],
        "learning_rate": [1e-4, 1e-5],
        "weight_decay": [1e-5],
        "optimizer":  ["RMSProp", "AdamW"],
        "scheduler": ["ReduceLROnPlateau"],
        "dataset": ["sampler"],
        "criterion": ["CrossEntropyLoss"],
        "seq_len": seq_len,
        "kmer": kmer,
        "embed": embed
    },
    # "RNN": {
    #     "batch_size": [64],
    #     "base_seed": [42],
    #     "learning_rate": [1e-4],
    #     "weight_decay": [1e-5],
    #     "optimizer": ["RMSProp"],
    #     "scheduler": ["ReduceLROnPlateau"],
    #     "dataset": ["sampler"],
    #     "criterion": ["CrossEntropyLoss"],
    #     "seq_len": seq_len,
    #     "kmer": kmer,
    #     "embed": embed
    # "1DCNN": {
    #     "batch_size": [16, 32],
    #     "base_seed": [42],
    #     "learning_rate": [1e-4, 1e-5],
    #     "weight_decay": [1e-5],
    #     "optimizer": ["RMSProp"],
    #     "scheduler": ["ReduceLROnPlateau"],
    #     "dataset": ["sampler"],
    #     "criterion": ["CrossEntropyLoss"],
    #     "seq_len": seq_len,
    #     "kmer": kmer,
    #     "embed": embed
    # },
    "1DCNN": {
        "batch_size": [16,32],
        "base_seed": [42],
        "learning_rate": [1e-4, 1e-5],
        "weight_decay": [1e-5],
        "optimizer": ["RMSProp", "AdamW"],
        "scheduler": ["ReduceLROnPlateau"],
        "dataset": ["sampler"],
        "criterion": ["CrossEntropyLoss"],
        "seq_len": seq_len,
        "kmer": kmer,
        "embed": embed
    },
    "Transformer": {
        "batch_size": [16, 32],
        "base_seed": [42],
        "learning_rate": [1e-4, 1e-5],
        "weight_decay": [1e-5],
        "optimizer": ["AdamW"],
        "scheduler": ["ReduceLROnPlateau"],
        "dataset": ["sampler"],
        "criterion": ["CrossEntropyLoss"],
        "seq_len": seq_len,
        "kmer": kmer,
        "embed": embed
    },
    # "Transformer": {
    #     "batch_size": [128],
    #     "base_seed": [42],
    #     "learning_rate": [1e-4],
    #     "weight_decay": [0, 1e-3],
    #     "optimizer": ["AdamW"],
    #     "scheduler": ["ReduceLROnPlateau"],
    #     "dataset": ["sampler"],
    #     "criterion": ["CrossEntropyLoss"],
    #     "seq_len": seq_len,
    #     "kmer": kmer,
    #     "embed": embed
    # },   
}



model_search_spaces = {
    # "RNN": {
    #     "embed_dim":   [64],
    #     "hidden_dim":   [128],
    #     "num_layers":   [2],
    #     "rnn_type":     ["GRU"],
    #     "bidirectional":[True],
    #     "dropout":      [0.1,0.2],
    #     "pooling":      ["max", "avg", "central", "central_attention", "attention"],
    #     "seq_len": seq_len,
    #     "kmer": kmer,
    #     "embed":  embed
    # },
    "RNN": {
        "embed_dim":   [64,128],
        "hidden_dim":   [128, 256, 384],
        "num_layers":   [1,2,3],
        "rnn_type":     ["GRU"],
        "bidirectional":[True],
        "dropout":      [0.1],
        "pooling":      ["average_attention"],
        "seq_len": seq_len,
        "kmer": kmer,
        "embed":  embed
    },
    # "1DCNN": {              
    #     "num_filters":[[32, 64, 128], [32, 64]],
    #     "kernel_sizes": [[3, 3, 3], [5, 5, 5], [3,3], [5,5]],
    #     "pool_sizes": [[1, 2, 2], [1,2]],
    #     "drop_out_rate": [0.2],
    #     "seq_len": seq_len,
    #     "kmer": kmer,
    #     "embed":  ["one_hot"],
    # },
    "1DCNN": {              
        "num_filters":[[32, 64, 128], [32, 64]],
        "kernel_sizes": [[3, 3, 3], [5, 5, 5], [3,3], [5,5]],
        "pool_sizes": [[1, 2, 2], [1,2]],
        "drop_out_rate": [0.2],
        "seq_len": seq_len,
        "kmer": kmer,
        "embed":  embed
    },
    # "Transformer": {
    #     "embed_dim": [200, 320, 480],
    #     "num_blocks": [3, 6, 12],
    #     "num_heads": [10],
    #     "head_type": ["max", "avg", "central", "central_attention", "attention"],
    #     "seq_len": seq_len,
    #     "kmer": kmer,
    #     "embed": ["one_hot"]
    # },
    "Transformer": {
        "embed_dim": [400, 600],
        "num_blocks": [2,4],
        "num_heads": [20],
        "head_type":  ["average_attention", "avg", "max"],
        "seq_len": seq_len,
        "kmer": kmer,
        "embed": embed
    }
}

def filter_rnn_cfg(cfg):
    return cfg["hidden_dim"] in {2*cfg["embed_dim"], 3*cfg["embed_dim"]}

def get_search_space(model_name):
    """
    This function returns the search space for a given model name.
    """
    if model_name not in model_search_spaces:
        raise ValueError(f"Model {model_name} not found in search spaces.")
    
    global_params = ParameterGrid(global_search_space[model_name])
    model_params = ParameterGrid(model_search_spaces[model_name])

    model_params_filtered = []    
    if model_name == "Transformer":
        for m in model_params:
            model_params_filtered.append(m)
    elif model_name == "1DCNN":
        for m in model_params:
            if len(m["num_filters"]) == len(m["kernel_sizes"]) and len(m["num_filters"]) == len(m["pool_sizes"]):
                model_params_filtered.append(m)
    elif model_name == "RNN":
        for m in model_params:
            if  filter_rnn_cfg(m):
                model_params_filtered.append(m) 
    else:
        raise ValueError(f"Model {model_name} not found in search spaces.")
    
    all_configs = list(itertools.product(global_params, model_params_filtered))

    all_configs_final = []
    for global_all, model_all in all_configs:
        if (global_all["seq_len"] == model_all["seq_len"]) and (global_all["kmer"] == model_all["kmer"]) and (global_all["embed"] == model_all["embed"]):
            all_configs_final.append((global_all, model_all))
    
    return all_configs_final


def model_mapping(model_name):
    if model_name == "1DCNN":
        from models.CNN1D import CNNClassifier
        return CNNClassifier
    elif model_name == "RNN":
        from models.RNNClassifier import RNNClassifier
        return RNNClassifier
    elif model_name ==  "Transformer":
        from models.Transformer_classifier import Transformer_classifier
        return Transformer_classifier
    elif model_name == "TransformerxCNN1D":
        from models.TransformerxCNN1D import TransformerxCNN1D
        return TransformerxCNN1D
    else:
        raise ValueError(f"Model {model_name} not found.")

