import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.optim

from utils.focal_loss import MultiClassFocalLoss
from utils.training_metrics import training_metrics
from utils.init_functions import init_func
from utils.seed     import set_global_seed


# Use relative dataset directory from repository root
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"



model_name = "Transformer"  # "1DCNN", "RNN", "Transformer"

if model_name == "RNN":
    sequence_len = 51
    kmer = 1
    batch_size = 16
    base_seed = 42
    set_global_seed(base_seed)
    max_epochs = 43
    learning_rate = 1e-5
    weight_decay = 1e-5
    optimizer_name = "AdamW"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "ReduceLROnPlateau"
    patience_scheduler = 4

elif model_name == "1DCNN":
    sequence_len = 51
    kmer = 1
    batch_size = 16
    base_seed = 42
    set_global_seed(base_seed)
    max_epochs = 50
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
    set_global_seed(base_seed)
    max_epochs = 23
    learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer_name = "AdamW"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "CosineAnnealingWarmRestarts"

 

criterion_name = "CrossEntropyLoss"
dataset_name = "sampler"  # "stratified", "sampler"
embed = "one_hot"  # "one_hot", "ENAC", "embeddings"


path_pickles = DATASET_DIR / "training_set.pickle"

with open(path_pickles, "rb") as f:
    train_dict = pickle.load(f)


output_main_dir = Path("./logs")
if not output_main_dir.exists():
    output_main_dir.mkdir()


output_dir = output_main_dir / (model_name + "_41nts")
output_dir.mkdir(exist_ok=True)


if criterion_name == "CrossEntropyLoss":
    criterion = nn.CrossEntropyLoss()
    
elif criterion_name == "FocalLoss":
    params_focal = {
        "alpha": 0.25,
        "gamma": 3.0,
        "reduction": "mean"
    }
    criterion = MultiClassFocalLoss(**params_focal)


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
    model = Transformer_classifier(**params_model)

        


if dataset_name == "stratified":
    from dataset_generation.dataset_train_stratified import Datasettrain as Datasettrain
elif dataset_name == "sampler":
    from dataset_generation.dataset_train_sampler import Datasettrain as Datasettrain



train_dataset = Datasettrain(
    data_dict=train_dict,
    base_seed=base_seed,
    embed=embed,
    kmer=kmer,
    seq_len=sequence_len,
)


if dataset_name == "sampler":
    train_dataset.set_weights()
    gen = torch.Generator()
    gen.manual_seed(base_seed)
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
    pin_memory=True
    )



output_columns = ["epoch", "train_loss", "train_f1", "train_precision", "train_recall", "train_auprc", "train_auprc_custom", "train_auroc", "train_accuracy"]
output_csv = output_dir / "output.csv"
output_df = pd.DataFrame(columns=output_columns, dtype=float)
output_df.to_csv(output_csv, index=False)


model.apply(init_func)

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
        "T_0": 8,
        "T_mult": 1, 
        "eta_min": 1e-7
    }
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **dict_scheduler)

elif scheduler_name == "StepLR":
    dict_scheduler = {
        "step_size": 10,
        "gamma": 0.1
    }
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **dict_scheduler)

elif scheduler_name == "CosineAnnealingLR":
    dict_scheduler = {
        "T_max": max_epochs,
        "eta_min": 1e-6
    }
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **dict_scheduler)

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

elif scheduler_name == "None":
    scheduler = None



if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU")
else:
    print("Using CPU")


for epoch in range(max_epochs):
 
    if dataset_name == "stratified":
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        

    logits_train = []
    labels_train = []
    loss_train = []
    input_ids_train = []
    for batch_index, batch in enumerate(train_loader):
        input_ids = batch["input_ids"]
        labels = batch["label"]

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()

        with torch.cuda.amp.autocast():
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)

            if scheduler_name == "CosineAnnealingWarmRestarts":
                scheduler.step(epoch + batch_index / len(train_loader))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_train.append(loss.detach().cpu().item())
        logits_train.append(logits.detach().cpu())
        labels_train.append(labels.detach().cpu())
        input_ids_train.append(input_ids.detach().cpu())
        
    loss_train = np.mean(loss_train)
    logits_train = torch.cat(logits_train, dim=0).float()
    labels_train = torch.cat(labels_train, dim=0)

    training_results, cm_mtx_training = training_metrics(logits_train, labels_train, loss_train, epoch=epoch+1)
    


    if scheduler_name in ["StepLR", "CosineAnnealingLR"]:
        scheduler.step()

    
    print(f"Epoch {epoch + 1}")
    
    if scheduler_name == "ReduceLROnPlateau":
        scheduler.step(training_results["train_auprc_custom"])


    new_row = pd.DataFrame([training_results])
    new_row = new_row.reindex(columns=output_df.columns)
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(output_csv, index=False)
    



weights_path = output_dir / f"{model_name}_epoch_{epoch + 1}.pt"
torch.save(model.state_dict(), weights_path)
# logits_train = logits_train.numpy()
# labels_train = labels_train.numpy()
# dict_train_res = {
#     "logits": logits_train,
#     "labels": labels_train,
#     "input_ids": input_ids_train,
# }
# #save training logits and labels in pickle
# with open(output_dir / f"train_logits_labels.pickle", "wb") as f:
#     pickle.dump(dict_train_res, f)
 
    






