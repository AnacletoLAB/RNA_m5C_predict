import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import pickle, json
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from utils.search_space import model_mapping
from utils.seed         import set_global_seed

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
RESULTS_DIR = Path(__file__).resolve() / "results"

from sklearn.metrics import confusion_matrix


def Inference(path_config, path_model, data_path, negatives= True, batch_size=256):
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(path_config, 'r') as f:
        config = json.load(f)

    model_name = config["model"]
    model_params = config["model_params"]

    seq_len = model_params["seq_len"]
    embed = model_params["embed"]
    kmer = model_params["kmer"]

    batch_size = 256
    seed = 42
    set_global_seed(seed)

    if not negatives:
        from dataset_generation.val_dataset_only_pos import Datasetval as Dataset_
    else:
        from dataset_generation.val_dataset import Datasetval as Dataset_

    dataset_ = Dataset_(
            data_dict=data,
            base_seed=seed,
            embed=embed,
            kmer=kmer,
            seq_len=seq_len,
        )

    dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)



    model = model_mapping(model_name)(**model_params)
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")


    all_input_ids = []
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].cuda() if torch.cuda.is_available() else batch["input_ids"]
                labels = batch["label"].cuda() if torch.cuda.is_available() else batch["label"]
                
                with torch.cuda.amp.autocast():
                    logits = model(input_ids)
                    preds = torch.argmax(logits, dim=1)
                    probs = F.softmax(logits, dim=1)
                    

                
                all_input_ids.append(input_ids.cpu())
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    idx = all_input_ids.argmax(dim=2) # → [7892, 51], values in {0,1,2,3}
    zero_mask = all_input_ids.sum(dim=2) == 0 # True wherever “P” is
    idx[zero_mask] = 4 # reserve index 4 for “P”

    charmap = np.array(list("ACGTP"), dtype="<U1")

    idx_np = idx.numpy()            # (7892, 51) integer array
    letters = charmap[idx_np]             # (7892, 51) array of '<U1'
    seqs = np.char.join("", letters)      # (7892,) array of full strings

    cm = confusion_matrix(all_labels, all_preds)

    results = {"seqs": seqs,
            "input_ids": all_input_ids.numpy(),
            "all_preds": all_preds,
            "all_probs": all_probs,
            "all_labels": all_labels,
            "confusion_matrix": cm,
            }
    

    return results

CURRENT_DIR = Path(__file__).parent

mapping_models = {
    "RNN": {"path_conf":CURRENT_DIR / "models_params/RNN_config.json",
            "path_weights":CURRENT_DIR / "logs/RNN/RNN_epoch_43.pt"},
    "Transformer": {"path_conf":CURRENT_DIR / "models_params/Transformer_config.json",
                    "path_weights": CURRENT_DIR / "logs/Transformer/Transformer_epoch_23.pt"},
    "1DCNN": {"path_conf":CURRENT_DIR / "models_params/1DCNN_config.json",
                    "path_weights": CURRENT_DIR / "logs/1DCNN/1DCNN_epoch_50.pt"},
}


if __name__ == "__main__":


    paths_data = {
        "test_set": DATASET_DIR / "test_set.pickle",
        "training_set": DATASET_DIR / "training_set.pickle",
    }

    results_all = {}
    cmx_all = {}
    for set_, path_data in paths_data.items():
        results_data = {}
        for model, paths in mapping_models.items():
            path_config = paths["path_conf"]
            path_weights = paths["path_weights"]

            results = Inference(path_config, path_weights, path_data)

            results_data[model] = results

        results_all[set_] = results_data



    output_path = RESULTS_DIR / "inference_results"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "inference_results.pickle", "wb") as f:
        pickle.dump(results_all, f)

