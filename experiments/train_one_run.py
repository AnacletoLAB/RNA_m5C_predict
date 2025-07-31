import os
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
from utils.init_functions import init_func

def training(global_params, model, train_dict, val_dict, output_dir, current_seed):

    patience = 10

    if global_params["dataset"] == "stratified":
        from dataset_generation.dataset_train_stratified import Datasettrain as Datasettrain
    elif global_params["dataset"] == "sampler":
        from dataset_generation.dataset_train_sampler import Datasettrain as Datasettrain

    if global_params["criterion"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
        
    elif global_params["criterion"] == "FocalLoss":
        params_focal = {
            "alpha": 0.25,
            "gamma": 3.0,
            "reduction": "mean"
        }
        criterion = MultiClassFocalLoss(**params_focal)


    from dataset_generation.val_dataset import Datasetval as Datasetval

    train_dataset = Datasettrain(
        data_dict=train_dict,
        base_seed=current_seed,
        embed=global_params["embed"],
        kmer=global_params["kmer"],
        seq_len=global_params["seq_len"],
    )
    val_dataset = Datasetval(
        data_dict=val_dict,
        base_seed=9999,
        embed=global_params["embed"],
        kmer=global_params["kmer"],
        seq_len=global_params["seq_len"],
    )

    if global_params["dataset"] == "sampler":
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
        batch_size=global_params["batch_size"],
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
        
    scaler = torch.cuda.amp.GradScaler()

    if global_params["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=global_params["learning_rate"], weight_decay=global_params["weight_decay"])
    elif global_params["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=global_params["learning_rate"],
            momentum=0.9,
            weight_decay=global_params["weight_decay"],
            nesterov=True
        )
    elif global_params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=global_params["learning_rate"],
            alpha=0.99,  # smoothing constant
            weight_decay=global_params["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {global_params['optimizer']}")

    if  global_params["scheduler"] == "CosineAnnealingWarmRestarts":
        dict_scheduler = {
            "T_0": 4,
            "T_mult": 1,
            "eta_min": 1e-7
        }
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **dict_scheduler)

    elif global_params["scheduler"] == "StepLR":
        dict_scheduler = {
            "step_size": 10,
            "gamma": 0.1
        }
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **dict_scheduler)

    elif global_params["scheduler"] == "CosineAnnealingLR":
        dict_scheduler = {
            "T_max": 300,
            "eta_min": 1e-6
        }
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **dict_scheduler)

    elif global_params["scheduler"] == "ReduceLROnPlateau":
        dict_scheduler = {
            "mode": "max",        # or 'min' for loss
            "factor": 0.5,        # halve LR
            "patience": 6,        # wait 4 epochs with no improvement
            "threshold": 1e-4,    # ignore tiny metric changes
            "min_lr": 1e-7,
            "verbose": False
        }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **dict_scheduler)

    elif global_params["scheduler"] == "None":
        scheduler = None


    if torch.cuda.is_available():
        model = model.cuda()
        #print("Using GPU")
    else:
        #print("Using CPU")
        pass

    output_columns = ["epoch", "val_loss", "val_f1", "val_precision", "val_recall", "val_auprc", "val_auprc_custom", "val_auroc", "val_accuracy"]
    output_csv = output_dir / "output.csv"
    output_df = pd.DataFrame(columns=output_columns, dtype=float)
    output_df.to_csv(output_csv, index=False)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    initial_validation, confusion_mtx = validation_metrics(model, nn.CrossEntropyLoss(), val_loader)
    #print("Initial validation:", initial_validation)

    best_results = initial_validation
    
    best_auprc = 0
    epochs_no_improve = 0

    for epoch in range(150):
        ##print minutes to do 1 epoch
        start = time.time()
    
        if global_params["dataset"] == "stratified":
            train_dataset.set_epoch(epoch)
            train_loader = DataLoader(train_dataset, batch_size=global_params["batch_size"], shuffle=False)
            

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

                if global_params["scheduler"] == "CosineAnnealingWarmRestarts":
                    scheduler.step(epoch + batch_index / len(train_loader))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


        if global_params["scheduler"] in ["StepLR", "CosineAnnealingLR"]:
            scheduler.step()
        
        ##print minutes to do 1 epoch (in minutes!)
        end = time.time()
        minutes = (end - start) / 60
        seconds = (end - start) % 60
        #print(f"Epoch {epoch} | Time: {minutes:.0f}m {seconds:.0f}s")
        validation_results, cm_mtx_val = validation_metrics(model, nn.CrossEntropyLoss(), val_loader, epoch+1)

        if global_params["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(validation_results["val_auprc_custom"])


        new_row = pd.DataFrame([validation_results])
        new_row = new_row.reindex(columns=output_df.columns)
        output_df = pd.concat([output_df, new_row], ignore_index=True)
        output_df.to_csv(output_csv, index=False)


        current_auprc = validation_results["val_auprc_custom"]
        if current_auprc > best_auprc:
            best_auprc = current_auprc
            epochs_no_improve = 0
            best_results = validation_results

            cm_path_val = output_dir / f"val_confusion_matrix_epoch_{epoch+1}.csv"

            #remove previous confusion matrix
            previous_cm_path = [x.name for x in output_dir.iterdir() if "confusion_matrix" in x.name]
            if len(previous_cm_path) > 0:
                (output_dir / previous_cm_path[0]).unlink()

            cm_mtx_val.to_csv(cm_path_val, index=False)

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                #print(f"No improvement in AUPRC for {patience} consecutive epochs. Stopping early.")
                break

    return best_results

