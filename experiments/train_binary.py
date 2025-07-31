import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim

from utils.training_metrics_binary import validation_metrics
from utils.training_metrics_binary import training_metrics
from utils.init_functions import init_func
from utils.seed     import set_global_seed
from utils.generate_data import generate_data



model_name = "Transformer"  # "1DCNN", "RNN", "Transformer"
which_data = "current_dataset"  # "mlm5c" or "current_dataset" or "deepm5C"
max_epochs = 200

if model_name == "RNN":
    kmer = 1
    batch_size = 16
    base_seed = 123
    learning_rate = 1e-5
    weight_decay = 1e-5
    optimizer_name = "AdamW"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "ReduceLROnPlateau"
    patience_scheduler = 4

elif model_name == "1DCNN":
    kmer = 1
    batch_size = 16
    base_seed = 123
    learning_rate = 1e-5
    weight_decay = 1e-5
    optimizer_name = "RMSProp"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "ReduceLROnPlateau"
    patience_scheduler = 6

elif model_name == "Transformer":
    kmer = 1
    batch_size = 32
    base_seed = 123
    learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer_name = "AdamW"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "ReduceLROnPlateau"
    patience_scheduler = 6


criterion_name = "CrossEntropyLoss"
embed = "one_hot"  # "one_hot", "ENAC", "embeddings"

set_global_seed(base_seed)

if which_data == "mlm5c":
    sequence_len = 201
elif which_data == "current_dataset":
    sequence_len = 151
elif which_data == "deepm5C":
    sequence_len = 41


df_train, df_val, df_test = generate_data(which_data=which_data, seed=base_seed)


output_main_dir = Path("./deep_runs_binary")
output_main_dir.mkdir(exist_ok=True, parents=True)


output_dir = output_main_dir / (f"{model_name}_{which_data}")
output_dir.mkdir(exist_ok=True, parents=True)


criterion = nn.CrossEntropyLoss()
    

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
                "num_classes": 2,
    }
    model = CNNClassifier(**params_model)

elif model_name == "RNN":
    assert embed != "embeddings", "RNN model does not support embeddings"
    from models.RNNClassifier import RNNClassifier
    params_model = {
                "embed_dim":128,
                "hidden_dim":256,
                "num_layers":3,
                "rnn_type":"GRU",
                "bidirectional":True,
                "dropout":0.1,
                "pooling":"central_attention",
                "seq_len": sequence_len,
                "kmer": kmer,
                "embed": embed,
                "num_classes": 2,
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
                    "num_classes": 2,
                    }
    model = Transformer_classifier(**params_model)
        
from dataset_generation.dataset_binary import Dataset_

train_dataset = Dataset_(
    df_=df_train,
    base_seed=base_seed,
    embed=embed,
    kmer=kmer,
    seq_len=sequence_len,
)
val_dataset = Dataset_(
    df_=df_val,
    base_seed=9999,
    embed=embed,
    kmer=kmer,
    seq_len=sequence_len,
)



weights_dir = output_dir / "weights"
if not weights_dir.exists():
    weights_dir.mkdir()

output_columns = ["epoch", "val_accuracy", "val_auroc", "val_auprc", "val_recall", "val_precision", "val_f1",
                  "train_accuracy", "train_auroc", "train_auprc", "train_recall", "train_precision", "train_f1"]
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
        "T_0": 4,
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

assert torch.cuda.is_available(), "CUDA is not available, but amp.autocast is used. Set scheduler_name to 'None' to disable it."

model = model.cuda()


val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
initial_validation = validation_metrics(model, val_loader, verbose=True)
print("Initial validation:", initial_validation)

#early stopping
best_metric = -float('inf')
epochs_no_improve = 0
patience = 10

for epoch in range(max_epochs):

    start = time.time()
 
    train_dataset.set_epoch(epoch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        

    logits_train = []
    labels_train = []
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

        logits_train.append(logits.detach().cpu())
        labels_train.append(labels.detach().cpu())
        
    logits_train = torch.cat(logits_train, dim=0).float()
    labels_train = torch.cat(labels_train, dim=0)

    training_results = training_metrics(logits_train, labels_train, epoch=epoch + 1)
    


    if scheduler_name in ["StepLR", "CosineAnnealingLR"]:
        scheduler.step()
    

    end = time.time()
    minutes = (end - start) / 60
    seconds = (end - start) % 60
    print(f"Epoch {epoch+1} | Time: {minutes:.0f}m {seconds:.0f}s")
    validation_results = validation_metrics(model, val_loader, epoch+1, verbose=True)

    if scheduler_name == "ReduceLROnPlateau":
        scheduler.step(validation_results["val_accuracy"])


    new_row = pd.DataFrame([dict(**validation_results, **training_results)])
    new_row = new_row.reindex(columns=output_df.columns)
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(output_csv, index=False)


    current_metric = validation_results["val_accuracy"]

    if current_metric > best_metric:
        best_metric = current_metric
        epochs_no_improve = 0
        model_name_dir = f"{epoch+1}_accuracy_{validation_results['val_accuracy']:.4f}_auroc_{validation_results['val_auroc']:.4f}.pt"
        previous_weight_files = [x.name for x in weights_dir.iterdir()]
        if len(previous_weight_files) > 0:
            previous_weight_file = previous_weight_files[0]
            (weights_dir / previous_weight_file).unlink()
        torch.save(model.state_dict(), weights_dir / model_name_dir)

    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"No improvement in wrecall for {patience} consecutive epochs. Stopping early.")
            
            break



#load the best model
model.load_state_dict(torch.load(weights_dir / model_name_dir))

if torch.cuda.is_available():
    model = model.cuda()


test_dataset = Dataset_(
    df_=df_test,
    base_seed=9999,
    embed=embed,
    kmer=kmer,
    seq_len=sequence_len,
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
cm, test_results = validation_metrics(model, test_loader, verbose=True, dataset_type="test")

test_results_df = pd.DataFrame([test_results])
test_results_df.to_csv(output_dir / "test_results.csv", index=False)
print(f"Test results saved to {output_dir / 'test_results.csv'}")

# Save confusion matrix plain
cm = pd.DataFrame(cm)
cm.to_csv(output_dir / "confusion_matrix.csv", index=False)






