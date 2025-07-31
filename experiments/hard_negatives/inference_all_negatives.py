import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

import sys
CURRENT_DIR = Path(__file__).parent
WORKING_DIR = CURRENT_DIR.parent
sys.path.append(str(WORKING_DIR))

from dataset_generation.dataset_inference_negatives import Dataset_negatives


from utils.search_space import model_mapping
from utils.seed         import set_global_seed




path_data = WORKING_DIR.parent / "dataset/all_remaining_negatives.pickle"

model_name = "RNN"
path_config = WORKING_DIR / f"models_params/{model_name}_config.json"

with open(path_data, 'rb') as f:
    data = pickle.load(f)

with open(path_config, 'r') as f:
    config = json.load(f)

model_params = config["model_params"]

seq_len = model_params["seq_len"]
embed = model_params["embed"]
kmer = model_params["kmer"]

model = model_mapping(model_name)(**model_params)


batch_size = 65536
seed = 42
set_global_seed(seed)

dataset_ = Dataset_negatives(
        list_sequences_negatives=data,
        base_seed=seed,
        embed=embed,
        kmer=kmer,
        seq_len=seq_len,
    )
dataloader = DataLoader(dataset_, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

dir_predictions = CURRENT_DIR / "res_negatives_predictions"
dir_predictions.mkdir(parents=True, exist_ok=True)

for fold in range(1, 6):

    path_model = CURRENT_DIR / f"results_first_training/fold_{fold}/weights/fold_{fold}.pt"
    model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
    model.eval()
    assert torch.cuda.is_available(), "CUDA is not available. Please check your setup."
    model = model.cuda()
    print("Using GPU")

    with torch.no_grad(), torch.cuda.amp.autocast():
        seqs_list = []
        all_probs = []
        all_preds = []
        for batch in dataloader:
            input_ids = batch
            logits = model(input_ids.cuda())
            preds = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

            idx_batch = input_ids.argmax(2).cpu()            # (batch, 51)
            zero_mask = input_ids.sum(2) == 0                # True where padding
            idx_batch[zero_mask] = 4                         # force to “P”

            letters = np.array(list("ACGTP"), dtype="<U1")[idx_batch.numpy()]
            seqs_batch = letters.view(f"<U{seq_len}").squeeze()   # (batch,)
            seqs_list.append(seqs_batch)


        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()

        seqs = np.concatenate(seqs_list, axis=0)  # Concatenate all sequences from the list


    results = {"seqs": seqs,
        "all_preds": all_preds,
        "all_probs": all_probs,
        }
        

    results_path = dir_predictions / f"fold_{fold}.pickle"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)


