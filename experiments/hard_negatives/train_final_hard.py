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
import sys
sys.path.append(str(WORKING_PATH))

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.optim


from utils.training_metrics import validation_metrics
from utils.training_metrics import training_metrics
from utils.init_functions import init_func
from utils.seed     import set_global_seed


# pi = [0.70, 0.075, 0.075, 0.075, 0.075]
pi = [0.83,0.0425,0.0425,0.0425,0.0425]
# pi = None
nsun2_fraction = 1.5

model_name = "RNN"  # "1DCNN", "RNN", "Transformer"


path_model = None  # Path to a pretrained model, if any. If None, the model will be initialized with random weights.

max_epoch_end = False
if max_epoch_end:
    max_epochs = 30
else:
    max_epochs = 300

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

if max_epoch_end:
    patience_scheduler = max_epochs


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
from dataset_generation.val_dataset import Datasetval as Datasetval


path_train_val = CURRENT_PATH / f"folds_augmented_final_hard/upper_bound_{upper_bound}_lower_bound_{lower_bound}"

if nsun2_fraction is None:
    train_dict_path = path_train_val / f"train_final_augmented_hard.pickle"
    with open(train_dict_path, "rb") as f:
        train_dict = pickle.load(f)

    val_dict_path = path_train_val / f"val_final_augmented_hard.pickle"
    with open(val_dict_path, "rb") as f:
        val_dict = pickle.load(f)

else:
    train_dict_path = path_train_val / f"train_final_augmented_hard_nsun2x{nsun2_fraction}.pickle"
    with open(train_dict_path, "rb") as f:
        train_dict = pickle.load(f)

    val_dict_path = path_train_val / f"val_final_augmented_hard_nsun2x{nsun2_fraction}.pickle"
    with open(val_dict_path, "rb") as f:
        val_dict = pickle.load(f)


if pi is None:
    pi_name = "none"
else:
    pi_name = str(pi).replace(" ", "").replace(".", "_")

if path_model is None:
    path_model_name = "none"
else:
    path_model_name = str(path_model).replace("/", "_")[-50:]

if max_epoch_end:
    output_dir = CURRENT_PATH / f"results_augmented_training_final_hard/upper_bound_{upper_bound}_lower_bound_{lower_bound}_pi_{pi_name}_pretrained_{path_model_name}_{model_name}_nsunfrac_{nsun2_fraction}_max_epoch_{max_epochs}"
else:
    output_dir = CURRENT_PATH / f"results_augmented_training_final_hard/upper_bound_{upper_bound}_lower_bound_{lower_bound}_pi_{pi_name}_pretrained_{path_model_name}_{model_name}_nsunfrac_{nsun2_fraction}"
output_dir.mkdir(exist_ok=True, parents=True)
same_neg_num = False






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


val_dataset = Datasetval(
data_dict=val_dict,
base_seed=9999,
embed=embed,
kmer=kmer,
seq_len=sequence_len,
)

train_dataset.set_weights(pi=pi)
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

output_columns = ["epoch", "val_loss", "val_f1", "val_precision", "val_recall", "val_auprc", "val_auprc_custom", "val_auroc", "val_accuracy"]
output_csv = output_dir / "output.csv"
output_df = pd.DataFrame(columns=output_columns, dtype=float)
output_df.to_csv(output_csv, index=False)




if path_model is not None:
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    with open(output_dir / "model_name.txt", "w") as f:
        f.write(f"{model_name} - {path_model.name}\n")
    print(f"Loaded model from {path_model}")
else:
    model.apply(init_func)
    with open(output_dir / "model_name.txt", "w") as f:
        f.write(f"{model_name} - not loaded, initialized with random weights\n")
    
    
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

        if scheduler_name == "CosineAnnealingWarmRestarts":
            scheduler.step(epoch + batch_index / len(train_loader))


    if scheduler_name in ["StepLR", "CosineAnnealingLR"]:
        scheduler.step()
    

    end = time.time()
    minutes = (end - start) / 60
    seconds = (end - start) % 60
    print(f"Epoch {epoch} | Time: {minutes:.0f}m {seconds:.0f}s")
    validation_results, cm_mtx_val = validation_metrics(model, nn.CrossEntropyLoss(), val_loader, epoch+1, verbose=True)

    if scheduler_name == "ReduceLROnPlateau":
        scheduler.step(validation_results["val_auprc_custom"])


    new_row = pd.DataFrame([validation_results])
    new_row = new_row.reindex(columns=output_df.columns)
    output_df = pd.concat([output_df, new_row], ignore_index=True)
    output_df.to_csv(output_csv, index=False)


    current_auprc = validation_results["val_auprc_custom"]
    if current_auprc > best_auprc:
        best_auprc = current_auprc
        epochs_no_improve = 0
        model_name = f"epoch_{epoch+1}.pt"
        previous_weight_files = [x.name for x in weights_dir.iterdir()]
        if len(previous_weight_files) > 0:
            previous_weight_file = previous_weight_files[0]
            (weights_dir / previous_weight_file).unlink()
        torch.save(model.state_dict(), weights_dir / model_name)
        cm_path_val = output_dir / f"val_confusion_matrix_epoch_{epoch+1}.csv"
        #remove previous confusion matrix
        previous_cm_path = [x.name for x in output_dir.iterdir() if "confusion_matrix" in x.name]
        if len(previous_cm_path) > 0:
            (output_dir / previous_cm_path[0]).unlink()

        cm_mtx_val.to_csv(cm_path_val, index=False)


    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            if not max_epoch_end:
                print(f"No improvement in AUPRC for {patience} consecutive epochs. Stopping early.")
                break

if max_epoch_end:
    model_name = f"epoch_{epoch+1}.pt"
    previous_weight_files = [x.name for x in weights_dir.iterdir()]
    if len(previous_weight_files) > 0:
        previous_weight_file = previous_weight_files[0]
        (weights_dir / previous_weight_file).unlink()
    torch.save(model.state_dict(), weights_dir / model_name)
    cm_path_val = output_dir / f"val_confusion_matrix_epoch_{epoch+1}.csv"
    #remove previous confusion matrix
    previous_cm_path = [x.name for x in output_dir.iterdir() if "confusion_matrix" in x.name]
    if len(previous_cm_path) > 0:
        (output_dir / previous_cm_path[0]).unlink()

    cm_mtx_val.to_csv(cm_path_val, index=False)





