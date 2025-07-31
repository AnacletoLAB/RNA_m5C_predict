import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
import numpy as np



def validation_metrics(model, criterion, dataloader, epoch=0, verbose=False):
    model.eval()
    val_loss_sum = 0.0
    val_step_count = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda() if torch.cuda.is_available() else batch["input_ids"]
            labels = batch["label"].cuda() if torch.cuda.is_available() else batch["label"]
            
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                ce_loss = criterion(logits, labels) 
                val_loss_sum += ce_loss.item()
                val_step_count += 1
                labels = labels.cpu()
                logits = logits.cpu()

            # Move to CPU for metric computation
            logits = logits.float().cpu()  # shape: (B, 5)
            labels = labels.float().cpu()  # shape: (B,)

            # Convert logits -> predicted classes by argmax
            preds = torch.argmax(logits, dim=1)  # shape: (B,)

            probs = F.softmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()  

    # Basic multi-class metrics
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)  # shape (5,5)
    if verbose:
        print("Confusion Matrix Val:\n", cm)

    # Save confusion matrix to CSV  <-- ADDED
    cm_df = pd.DataFrame(cm)


    # Macro AUPRC & AUROC in one-vs-rest fashion  <-- ADDED
    # Binarize true labels for multi-class
    # e.g., label_binarize([0,2,4], classes=[0,1,2,3,4]) -> [[1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]]
    labels_bin = label_binarize(all_labels, classes=[0,1,2,3,4])  # shape (N,5)

    auprc = average_precision_score(labels_bin, all_probs, average="macro")
    auprc_custom = average_precision_score(labels_bin, all_probs, average=None)

    weights  = np.array([0.5] + [0.125]*4)
    auprc_custom = np.sum(auprc_custom * weights)
    auroc = roc_auc_score(labels_bin, all_probs, average="macro")

    val_loss = val_loss_sum / val_step_count
    if verbose:
        print(f"Epoch {epoch} | Val loss: {val_loss:.4f}")
        print(f"Epoch {epoch} | Val f1 (macro): {f1:.4f}")
        print(f"Epoch {epoch} | Val precision (macro): {precision:.4f}")
        print(f"Epoch {epoch} | Val recall (macro): {recall:.4f}")
        print(f"Epoch {epoch} | Val auprc (macro): {auprc:.4f}")
        print(f"Epoch {epoch} | Val auprc (custom): {auprc_custom:.4f}")
        print(f"Epoch {epoch} | Val auroc (macro): {auroc:.4f}")
        print(f"Epoch {epoch} | Val accuracy: {accuracy:.4f}")

    model.train()
    return {
        "epoch": epoch,
        "val_loss": val_loss,
        "val_f1": f1,
        "val_precision": precision,
        "val_recall": recall,
        "val_auprc": auprc,
        "val_auprc_custom": auprc_custom,
        "val_auroc": auroc,
        "val_accuracy": accuracy
    }, cm_df



def training_metrics(all_logits, all_labels, train_loss, epoch=0):

    all_preds = torch.argmax(all_logits, dim=1).numpy()
    all_probs = F.softmax(all_logits, dim=1).numpy()
    all_labels = all_labels.float().numpy()

    # Basic multi-class metrics
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)  # shape (5,5)
    print("Confusion Matrix Train:\n", cm)

    # Save confusion matrix to CSV  <-- ADDED
    cm_df = pd.DataFrame(cm)


    # Macro AUPRC & AUROC in one-vs-rest fashion  <-- ADDED
    # Binarize true labels for multi-class
    # e.g., label_binarize([0,2,4], classes=[0,1,2,3,4]) -> [[1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]]
    labels_bin = label_binarize(all_labels, classes=[0,1,2,3,4])  # shape (N,5)

    auprc = average_precision_score(labels_bin, all_probs, average="macro")
    auprc_custom = average_precision_score(labels_bin, all_probs, average=None)
    weights  = np.array([0.5] + [0.125]*4)
    auprc_custom = np.sum(auprc_custom * weights)
    auroc = roc_auc_score(labels_bin, all_probs, average="macro")


    print(f"Epoch {epoch} | Train loss: {train_loss:.4f}")
    print(f"Epoch {epoch} | Train f1 (macro): {f1:.4f}")
    print(f"Epoch {epoch} | Train precision (macro): {precision:.4f}")
    print(f"Epoch {epoch} | Train recall (macro): {recall:.4f}")
    print(f"Epoch {epoch} | Train auprc (macro): {auprc:.4f}")
    print(f"Epoch {epoch} | Train auprc (custom): {auprc_custom:.4f}")
    print(f"Epoch {epoch} | Train auroc (macro): {auroc:.4f}")
    print(f"Epoch {epoch} | Train accuracy: {accuracy:.4f}")
    print("-------------------------------------------------")

    return {
        "train_loss": train_loss,
        "train_f1": f1,
        "train_precision": precision,
        "train_recall": recall,
        "train_auprc": auprc,
        "train_auprc_custom": auprc_custom,
        "train_auroc": auroc,
        "train_accuracy": accuracy
    }, cm_df







def test_metrics(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda() if torch.cuda.is_available() else batch["input_ids"]
            labels = batch["label"].cuda() if torch.cuda.is_available() else batch["label"]
            
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                labels = labels.cpu()
                logits = logits.cpu()

            # Move to CPU for metric computation
            logits = logits.float().cpu()  # shape: (B, 5)
            labels = labels.float().cpu()  # shape: (B,)

            # Convert logits -> predicted classes by argmax
            preds = torch.argmax(logits, dim=1)  # shape: (B,)

            probs = F.softmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()  

    # Basic multi-class metrics
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)  # shape (5,5)

    # Save confusion matrix to CSV  <-- ADDED
    cm_df = pd.DataFrame(cm)


    # Macro AUPRC & AUROC in one-vs-rest fashion  <-- ADDED
    # Binarize true labels for multi-class
    # e.g., label_binarize([0,2,4], classes=[0,1,2,3,4]) -> [[1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]]
    labels_bin = label_binarize(all_labels, classes=[0,1,2,3,4])  # shape (N,5)

    auprc = average_precision_score(labels_bin, all_probs, average="macro")
    auprc_custom = average_precision_score(labels_bin, all_probs, average=None)

    weights  = np.array([0.5] + [0.125]*4)
    auprc_custom = np.sum(auprc_custom * weights)
    auroc = roc_auc_score(labels_bin, all_probs, average="macro")

    model.train()
    return {
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_auprc": auprc,
        "test_auprc_custom": auprc_custom,
        "test_auroc": auroc,
        "test_accuracy": accuracy
    }, cm_df