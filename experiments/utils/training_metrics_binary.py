import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import numpy as np



def validation_metrics(model, dataloader, epoch=0, verbose=False, dataset_type="val"):
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

            logits = logits.float().cpu()  # shape: (B, 2)
            labels = labels.float().cpu()  # shape: (B,)

            preds = torch.argmax(logits, dim=1)  # shape: (B,)

            probs = F.softmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()  


    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs[:, 1])
    auprc = average_precision_score(all_labels, all_probs[:, 1])
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    



    if verbose:
        print(f"Epoch {epoch} | Val accuracy: {accuracy:.4f}")
        print(f"Epoch {epoch} | Val AUROC: {auroc:.4f}")
        print(f"Epoch {epoch} | Val AUPRC: {auprc:.4f}")
        print(f"Epoch {epoch} | Val Recall: {recall:.4f}")
        print(f"Epoch {epoch} | Val Precision: {precision:.4f}")
        print(f"Epoch {epoch} | Val F1: {f1:.4f}")



    model.train()
    final_res = {
        "epoch": epoch,
        f"{dataset_type}_accuracy": accuracy,
        f"{dataset_type}_auroc": auroc,
        f"{dataset_type}_auprc": auprc,
        f"{dataset_type}_recall": recall,
        f"{dataset_type}_precision": precision,
        f"{dataset_type}_f1": f1,
    }
    if dataset_type == "val":
        return final_res

    cm = confusion_matrix(all_labels, all_preds)

    return cm, final_res



def training_metrics(all_logits, all_labels, epoch=0):

    all_preds = torch.argmax(all_logits, dim=1).numpy()
    all_probs = F.softmax(all_logits, dim=1).numpy()
    all_labels = all_labels.float().numpy()



    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs[:, 1])
    auprc = average_precision_score(all_labels, all_probs[:, 1])
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)



    print(f"Epoch {epoch} | Train accuracy: {accuracy:.4f}")
    print(f"Epoch {epoch} | Train AUROC: {auroc:.4f}")
    print(f"Epoch {epoch} | Train AUPRC: {auprc:.4f}")
    print(f"Epoch {epoch} | Train Recall: {recall:.4f}")
    print(f"Epoch {epoch} | Train Precision: {precision:.4f}")
    print(f"Epoch {epoch} | Train F1: {f1:.4f}")
    print("-------------------------------------------------")


    return {
        "train_accuracy": accuracy,
        "train_auroc": auroc,
        "train_auprc": auprc,
        "train_recall": recall,
        "train_precision": precision,
        "train_f1": f1,
    }


