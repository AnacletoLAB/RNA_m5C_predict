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
from utils.training_metrics import validation_metrics
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
    max_epochs = 300
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
    max_epochs = 300
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
    max_epochs = 300
    learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer_name = "AdamW"  # "AdamW", "SGD", "RMSProp"
    scheduler_name = "ReduceLROnPlateau"
    patience_scheduler = 6


criterion_name = "CrossEntropyLoss"

dataset_name = "sampler"  # "stratified", "sampler"
embed = "one_hot"  # "one_hot", "ENAC", "embeddings"


path_pickles = DATASET_DIR / "folds"
fold_number = 1
hard_negatives = False
only_hard = False
soft = True
lower_bound = 0.0
upper_bound = 0.6



if hard_negatives:
    if only_hard:
        path_train = Path(__file__).parent / f"results/negatives/fold_{fold_number}_{model_name}_hard_negatives_only.pickle"
    else:
        if not soft:
            path_train = Path(__file__).parent / f"results/negatives/fold_{fold_number}_{model_name}_hard_negatives_augmented.pickle"
        else:
            path_train = Path(__file__).parent / f"results/negatives/fold_{fold_number}_{model_name}_hard_negatives_augmented_soft_lb_{lower_bound}_ub_{upper_bound}.pickle"
    with open(path_train, "rb") as f:
        train_dict = pickle.load(f)
else:
    path_train = path_pickles / f"fold_{fold_number}_train.pickle"
    with open(path_train, "rb") as f:
        train_dict = pickle.load(f)


path_val = path_pickles / f"fold_{fold_number}_val.pickle"
with open(path_val, "rb") as f:
    val_dict = pickle.load(f)


load_path = "None"

output_main_dir = Path("./logss")
if not output_main_dir.exists():
    output_main_dir.mkdir()

output_subdirs = [int(x.name) for x in output_main_dir.iterdir() if x.is_dir()]
if len(output_subdirs) == 0:
    version = 1
else:
    version = max(output_subdirs) + 1
output_dir = output_main_dir / str(version)
output_dir.mkdir()

#save a txt with training path
with open(output_dir / "training_path.txt", "w") as f:
    f.write(str(path_train))
    if load_path != "None":
        f.write("\n")
        f.write(str(load_path))

params = {
    "batch_size": batch_size,
    "base_seed": base_seed,
    "max_epochs": max_epochs,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "optimizer": optimizer_name,
    "scheduler": scheduler_name,
    "load_path": load_path,
    "dataset": dataset_name,
    "model": model_name,
    "criterion": criterion_name,
    "seq_len": sequence_len,
    "kmer": kmer,
    "embed": embed
}


if criterion_name == "CrossEntropyLoss":
    criterion = nn.CrossEntropyLoss()
    
elif criterion_name == "FocalLoss":
    params_focal = {
        "alpha": 0.25,
        "gamma": 3.0,
        "reduction": "mean"
    }
    params["focal_loss"] = params_focal
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

        


elif model_name == "TransformerxCNN1D":
    from models.TransformerxCNN1D import TransformerxCNN1D
    params_model = {"num_blocks": 2,
                    "num_heads": 8,
                    "head_type": "central_attention",
                    "seq_len": sequence_len,
                    "kmer": kmer,
                    "embed": embed,
                    "num_filters":[32, 64, 128],
                    "kernel_sizes": [3, 3, 3],
                    "pool_sizes": [1, 1, 1],
                    "do_dropout": True,
                    "drop_out_rate": 0.2,
                    }
    params["TransformerxCNN1D"] = params_model
    model = TransformerxCNN1D(**params_model)

elif model_name == "minimal_Attention":
    from models.minimal_attention import minimal_Attention
    params_model = {"seq_len": sequence_len,
                    "c_in": 300,
                    "num_heads": 10,
                    }
    params["minimal_Attention"] = params_model
    model = minimal_Attention(**params_model)


if dataset_name == "stratified":
    from dataset_generation.dataset_train_stratified import Datasettrain as Datasettrain

    train_dataset = Datasettrain(
    data_dict=train_dict,
    base_seed=base_seed,
    embed=embed,
    kmer=kmer,
    seq_len=sequence_len,
    )

elif dataset_name == "sampler":
    from dataset_generation.dataset_train_sampler import Datasettrain as Datasettrain
    
    train_dataset = Datasettrain(
    data_dict=train_dict,
    base_seed=base_seed,
    embed=embed,
    kmer=kmer,
    seq_len=sequence_len,
    same_neg_num=False
    )


from dataset_generation.val_dataset import Datasetval as Datasetval


val_dataset = Datasetval(
    data_dict=val_dict,
    base_seed=9999,
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


weights_dir = output_dir / "weights"
if not weights_dir.exists():
    weights_dir.mkdir()

output_columns = ["epoch", "val_loss", "val_f1", "val_precision", "val_recall", "val_auprc", "val_auprc_custom", "val_auroc", "val_accuracy",
                  "train_loss", "train_f1", "train_precision", "train_recall", "train_auprc", "train_auprc_custom", "train_auroc", "train_accuracy"]
output_csv = output_dir / "output.csv"
output_df = pd.DataFrame(columns=output_columns, dtype=float)
output_df.to_csv(output_csv, index=False)


if load_path == "None":
    model.apply(init_func)
else:
    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))


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


if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU")
else:
    print("Using CPU")


val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
initial_validation, confusion_mtx = validation_metrics(model, nn.CrossEntropyLoss(), val_loader, verbose=True)
print("Initial validation:", initial_validation)

#early stopping
best_auprc = -float('inf')
epochs_no_improve = 0
patience = 10

for epoch in range(max_epochs):
    #print minutes to do 1 epoch
    start = time.time()
 
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

    training_results, cm_mtx_training = training_metrics(logits_train, labels_train, loss_train, epoch=epoch)
    


    if scheduler_name in ["StepLR", "CosineAnnealingLR"]:
        scheduler.step()
    
    #print minutes to do 1 epoch (in minutes!)
    end = time.time()
    minutes = (end - start) / 60
    seconds = (end - start) % 60
    print(f"Epoch {epoch} | Time: {minutes:.0f}m {seconds:.0f}s")
    validation_results, cm_mtx_val = validation_metrics(model, nn.CrossEntropyLoss(), val_loader, epoch+1, verbose=True)

    if scheduler_name == "ReduceLROnPlateau":
        scheduler.step(validation_results["val_auprc_custom"])


    new_row = pd.DataFrame([dict(**validation_results, **training_results)])
    new_row = new_row.reindex(columns=output_df.columns)
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(output_csv, index=False)


    current_auprc = validation_results["val_auprc_custom"]
    if current_auprc > best_auprc:
        best_auprc = current_auprc
        epochs_no_improve = 0
        model_name = f"{epoch+1}_f1_{validation_results['val_f1']:.4f}_auprc_{validation_results['val_auprc_custom']:.4f}.pt"
        previous_weight_files = [x.name for x in weights_dir.iterdir()]
        if len(previous_weight_files) > 0:
            previous_weight_file = previous_weight_files[0]
            (weights_dir / previous_weight_file).unlink()
        torch.save(model.state_dict(), weights_dir / model_name)
        cm_path_val = output_dir / f"val_confusion_matrix_epoch_{epoch}.csv"
        cm_path_train = output_dir / f"train_confusion_matrix_epoch_{epoch}.csv"
        #remove previous confusion matrix
        previous_cm_path = [x.name for x in output_dir.iterdir() if "confusion_matrix" in x.name]
        if len(previous_cm_path) > 0:
            (output_dir / previous_cm_path[0]).unlink()
            (output_dir / previous_cm_path[1]).unlink()

        cm_mtx_val.to_csv(cm_path_val, index=False)
        cm_mtx_training.to_csv(cm_path_train, index=False)
        #save training logits and labels in dict
        logits_train = logits_train.numpy()
        labels_train = labels_train.numpy()
        dict_train_res = {
            "logits": logits_train,
            "labels": labels_train,
            "input_ids": input_ids_train,
        }
        #save training logits and labels in pickle
        with open(output_dir / f"train_logits_labels.pickle", "wb") as f:
            pickle.dump(dict_train_res, f)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"No improvement in AUPRC for {patience} consecutive epochs. Stopping early.")
            
            break


    






