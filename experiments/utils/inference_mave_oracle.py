from pathlib import Path
import pandas as pd, numpy as np
import os, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

cwd = Path(os.getcwd())
WORKING_DIR = cwd.parent
sys.path.append(str(WORKING_DIR))
TRANSCRIPTOME_DIR = WORKING_DIR / "transcriptome_analysis/transcriptome"
path_premavenn = WORKING_DIR / "results/premavenn"
path_premavenn.mkdir(parents=True, exist_ok=True)
model_name = "RNN"

from inference import Inference
from inference import mapping_models

seed=42
np.random.seed(seed)

mapping_names = {
    "prob1": "type_1",
    "prob2": "type_2",
    "prob3": "type_3",
    "prob4": "type_4",}

res = TRANSCRIPTOME_DIR / "predictions_hard_mining.parquet"
preds = pd.read_parquet(res)
preds["max"] =  preds[["prob0", "prob1", "prob2", "prob3", "prob4"]].idxmax(axis=1)


tx = pd.read_csv(TRANSCRIPTOME_DIR / "gencode_customized.tsv", sep="\t")
tx["ids"] = tx.index


LEN = 51
KEEP = int(LEN / 2)
mutation_rate = 0.1
number_to_select = 1000000

mutated = {}
for prob_name, class_name in mapping_names.items():

    subset = preds[preds["max"] == prob_name]
    seqs = []
    for row in subset.itertuples():
        tx_idx, cpos = int(row.tx_idx), int(row.center_pos)
        seq = tx.iloc[tx_idx].sequence[cpos-KEEP : cpos+KEEP+1]
        if 'P' not in seq and len(seq) == LEN:
            seqs.append(seq)
    print(f"Number of sequences: {len(seqs)}")
    if len(seqs) > number_to_select:
        seqs_selected = np.random.choice(seqs, size=number_to_select, replace=False)
    else:
        seqs_selected = seqs.copy()
    multiples = int(np.ceil(number_to_select/len(seqs)))
    full = seqs_selected.copy() 
    for _ in range(multiples - 1):
        full.extend(seqs_selected)
    full = full[:number_to_select]
    print(f"Number of sequences after multiples: {len(full)}")
    full = np.array([list(x) for x in full])
    mutated_pos = np.random.poisson(lam=mutation_rate, size=(number_to_select, 51)).astype(bool)
    mutations = np.random.choice(("T", "C", "A", "G"), size=(number_to_select, 51))
    full[mutated_pos] = mutations[mutated_pos]
    full[:, LEN//2] = "C"
    full = ["".join(x) for x in full]

    mutated[class_name] = full.copy() + seqs.copy()
    print(f"Number of sequences final: {len(mutated[class_name])}")


path_premavenn_sequences = path_premavenn / f"mutations_{model_name}_mutation_rate_{mutation_rate}_number_mutated_{number_to_select}"
path_premavenn_sequences.mkdir(parents=True, exist_ok=True)
with open(path_premavenn_sequences / "mutated_sequences.pickle", "wb") as f:
    pickle.dump(mutated, f)


path_config = mapping_models[model_name]["path_conf"]
path_weights_dir = WORKING_DIR / "model_weights/heavy_hard_mining_RNN"
path_weights = [x for x in path_weights_dir.iterdir() if x.is_file() and x.suffix == ".pt"][0]
path_data = path_premavenn_sequences / "mutated_sequences.pickle"
results_inference = Inference(path_config, path_weights, path_data, negatives=False, batch_size=1024)

with open(path_premavenn_sequences / "inference_results.pickle", "wb") as f:
    pickle.dump(results_inference, f)