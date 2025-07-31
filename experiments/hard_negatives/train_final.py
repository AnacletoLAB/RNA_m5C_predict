import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--upper", type=float, required=False, default=1.0,
                    help="upper-bound threshold for the hard-negative window")
parser.add_argument("--lower", type=float, required=False, default=0.0,
                    help="lower-bound threshold for the hard-negative window")

args = parser.parse_args()



upper_bound = args.upper
lower_bound = args.lower


CURRENT_PATH = Path(__file__).parent
WORKING_PATH = CURRENT_PATH.parent
DATASET_DIR = WORKING_PATH.parent / "dataset"
import sys
sys.path.append(str(WORKING_PATH))

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.optim


from utils.training_metrics import training_metrics
from utils.init_functions import init_func
from utils.seed     import set_global_seed



model_name = "RNN"  # "1DCNN", "RNN", "Transformer"

if model_name == "RNN":
    sequence_len = 51
    kmer = 1
    batch_size = 16
    base_seed = 42
    learning_rate = 1e-5
    weight_decay = 1e-5
    optimizer_name = "AdamW"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "ReduceLROnPlateau"
    patience_scheduler = 4
    max_epochs = 13

elif model_name == "1DCNN":
    sequence_len = 51
    kmer = 1
    batch_size = 16
    base_seed = 42
    learning_rate = 1e-5
    weight_decay = 1e-5
    optimizer_name = "RMSProp"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "ReduceLROnPlateau"
    patience_scheduler = 6

elif model_name == "Transformer":
    sequence_len = 51
    kmer = 1
    batch_size = 32
    base_seed = 42
    learning_rate = 1e-5
    weight_decay = 1e-5
    optimizer_name = "AdamW"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "ReduceLROnPlateau"
    patience_scheduler = 6


criterion = nn.CrossEntropyLoss()

dataset_name = "sampler"  # "stratified", "sampler"
embed = "one_hot"  # "one_hot", "ENAC", "embeddings"


params = {
    "batch_size": batch_size,
    "base_seed": base_seed,
    "max_epochs": max_epochs,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "optimizer": optimizer_name,
    "scheduler": scheduler_name,
    "dataset": dataset_name,
    "model": model_name,
    "seq_len": sequence_len,
    "kmer": kmer,
    "embed": embed
}


if model_name == "1DCNN":
    assert embed != "embeddings", "1DCNN model does not support embeddings"
    from models.CNN1D import CNNClassifier
    params_model = {
                "num_filters":[32, 64],
                "kernel_sizes": [5, 5],
                "pool_sizes": [1, 2],
                "drop_out_rate": 0.2,
                "seq_len": sequence_len,
                "kmer": kmer,
                "embed": embed,
    }
    params["1DCNN"] = params_model
    model = CNNClassifier(**params_model)

elif model_name == "RNN":
    assert embed != "embeddings", "RNN model does not support embeddings"
    from models.RNNClassifier import RNNClassifier
    params_model = {
                "embed_dim":128,
                "hidden_dim":256,
                "num_layers":3,
                "num_classes":5,
                "rnn_type":"GRU",
                "bidirectional":True,
                "dropout":0.1,
                "pooling":"central_attention",
                "seq_len": sequence_len,
                "kmer": kmer,
                "embed": embed,
            }
    params["RNN"] = params_model
    model = RNNClassifier(**params_model)

elif model_name == "Transformer":
    from models.Transformer_classifier import Transformer_classifier
    params_model = {"embed_dim": 600,
                    "num_blocks": 4,
                    "num_heads": 20,
                    "head_type": "average_attention",
                    "seq_len": sequence_len,
                    "kmer": kmer,
                    "embed": embed,
                    }
    params["transformer"] = params_model
    model = Transformer_classifier(**params_model)

        

from dataset_generation.dataset_train_sampler import Datasettrain as Datasettrain


train_dict_path = CURRENT_PATH / f"folds_augmented_final/upper_bound_{upper_bound}_lower_bound_{lower_bound}/train_final_augmented.pickle"
output_dir = CURRENT_PATH / f"results_augmented_training_final/upper_bound_{upper_bound}_lower_bound_{lower_bound}_final"
output_dir.mkdir(exist_ok=True, parents=True)
same_neg_num = False



with open(train_dict_path, "rb") as f:
    train_dict = pickle.load(f)

current_seed = base_seed
set_global_seed(current_seed)
train_dataset = Datasettrain(
data_dict=train_dict,
base_seed=current_seed,
embed=embed,
kmer=kmer,
seq_len=sequence_len,
same_neg_num=same_neg_num
)



train_dataset.set_weights()
gen = torch.Generator()
gen.manual_seed(current_seed)
sampler = WeightedRandomSampler(
weights=train_dataset.weights,
num_samples=len(train_dataset),
replacement=True,
generator=gen)
train_loader = DataLoader(
train_dataset,
batch_size=batch_size,
sampler=sampler,
num_workers=2,
pin_memory=True)


weights_dir = output_dir / "weights"
if not weights_dir.exists():
    weights_dir.mkdir()

output_columns = ["epoch", "train_loss", "train_f1", "train_precision", "train_recall", "train_auprc", "train_auprc_custom", "train_auroc", "train_accuracy"]
output_csv = output_dir / "output.csv"
output_df = pd.DataFrame(columns=output_columns, dtype=float)
output_df.to_csv(output_csv, index=False)



path_model = WORKING_PATH / f"logs/{model_name}"
path_model = [x for x in path_model.iterdir() if x.name.startswith(model_name) and x.suffix == ".pt"][0]
model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))

    
scaler = torch.cuda.amp.GradScaler()

if optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "SGD":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
    )
elif optimizer_name == "RMSProp":
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        alpha=0.99,  # smoothing constant
        weight_decay=weight_decay
    )
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

if scheduler_name == "CosineAnnealingWarmRestarts":
    dict_scheduler = {
        "T_0": 4,
        "T_mult": 1,
        "eta_min": 1e-7
    }
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **dict_scheduler)
    params["scheduler_cosineanningwarmrestarts"] = dict_scheduler
elif scheduler_name == "StepLR":
    dict_scheduler = {
        "step_size": 10,
        "gamma": 0.1
    }
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **dict_scheduler)
    params["scheduler_steplr"] = dict_scheduler
elif scheduler_name == "CosineAnnealingLR":
    dict_scheduler = {
        "T_max": max_epochs,
        "eta_min": 1e-6
    }
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **dict_scheduler)
    params["scheduler_cosineanninglr"] = dict_scheduler
elif scheduler_name == "ReduceLROnPlateau":
    dict_scheduler = {
        "mode": "max",        # or 'min' for loss
        "factor": 0.5,        # halve LR
        "patience": patience_scheduler,        # wait 4 epochs with no improvement
        "threshold": 1e-4,    # ignore tiny metric changes
        "min_lr": 1e-7,
        "verbose": True
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **dict_scheduler)
    params["scheduler_reducelronplateau"] = dict_scheduler
elif scheduler_name == "None":
    scheduler = None


print("Params:", params)

with open(output_dir / "params.json", "w") as f:
    json.dump(params, f)


assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation."
model = model.cuda()
print("Using GPU")




for epoch in range(max_epochs):
    #print minutes to do 1 epoch
    start = time.time()
    
    logits_train = []
    labels_train = []
    loss_train = []
    for batch_index, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].cuda()
        labels = batch["label"].cuda()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():

            logits = model(input_ids)
            loss = criterion(logits, labels)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_train.append(loss.detach().cpu().item())
        logits_train.append(logits.detach().cpu())
        labels_train.append(labels.detach().cpu())

        if scheduler_name == "CosineAnnealingWarmRestarts":
            scheduler.step(epoch + batch_index / len(train_loader))


    if scheduler_name in ["StepLR", "CosineAnnealingLR"]:
        scheduler.step()


    loss_train = np.mean(loss_train)
    logits_train = torch.cat(logits_train, dim=0).float()
    labels_train = torch.cat(labels_train, dim=0)

    training_results, cm_mtx_training = training_metrics(logits_train, labels_train, loss_train, epoch=epoch+1)
    
    
    if scheduler_name == "ReduceLROnPlateau":
        scheduler.step(training_results["train_auprc_custom"])

    end = time.time()
    minutes = (end - start) / 60
    seconds = (end - start) % 60
    print(f"Epoch {epoch} | Time: {minutes:.0f}m {seconds:.0f}s")


    new_row = pd.DataFrame([training_results])
    new_row = new_row.reindex(columns=output_df.columns)
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(output_csv, index=False)


weights_path = output_dir / f"weights/{model_name}_epoch_{epoch + 1}.pt"
weights_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), weights_path)


