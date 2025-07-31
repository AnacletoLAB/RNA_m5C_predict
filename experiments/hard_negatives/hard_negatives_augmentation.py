import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import copy

seed = 42

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--uppers", type=float, nargs="+",
    default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    help="one or more upper-bound thresholds")
parser.add_argument(
    "--lowers", type=float, nargs="+",
    default=[0.0],
    help="one or more lower-bound thresholds")

args = parser.parse_args()

upper_bounds = args.uppers
lower_bounds = args.lowers


CURRENT_PATH = Path(__file__).parent
WORKING_DIR = CURRENT_PATH.parent
DATASET_DIR = WORKING_DIR.parent / "dataset"

res_negatives_prediction = CURRENT_PATH / "res_negatives_predictions"

path_initial_folds = DATASET_DIR / "folds"

folds_augmented = CURRENT_PATH / "folds_augmented"
folds_augmented.mkdir(exist_ok=True)


full_dataset = DATASET_DIR / "full_dataset_final_training.pickle"
with open(full_dataset, "rb") as f:
    full_data = pickle.load(f)


origin_seed_index = [defaultdict(list), defaultdict(list)]
origin_centres = []
for _, value in full_data.items():
    for seq in value:

        c21 = seq[65:151-65]
        origin_centres.append(c21)
        origin_seed_index[0][c21[:10]].append(len(origin_centres)-1)
        origin_seed_index[1][c21[11:]].append(len(origin_centres)-1)



def too_similar(seq51: str,
                centres: list[str],
                idx_lut: list[dict[str, list[int]]],
                max_mm: int = 2) -> bool:
    c21 = seq51[15:36]
    cand = set(idx_lut[0].get(c21[:10], [])) | set(idx_lut[1].get(c21[11:], []))
    for j in cand:
        if sum(a != b for a, b in zip(c21, centres[j])) <= max_mm:
            return True
    return False

all_idx_added = {}
for lower_bound in lower_bounds:
    for upper_bound in upper_bounds:
        idx_added_dict = {}
        path_threshold = folds_augmented / f"upper_bound_{upper_bound}_lower_bound_{lower_bound}"
        path_threshold.mkdir(exist_ok=True)

        for fold in range(1,6):
            idx_added_fold = defaultdict(list)
            centres = origin_centres.copy()
            seed_index = [copy.deepcopy(d) for d in origin_seed_index]

            rng_fold = np.random.default_rng(seed + fold)

            with open(res_negatives_prediction / f"fold_{fold}.pickle", "rb") as f:
                negatives = pickle.load(f)

            all_seqs = negatives["seqs"]
            all_preds = negatives["all_preds"]
            all_probs = negatives["all_probs"]

            path_fold = path_initial_folds / f"fold_{fold}_train.pickle"

            with open(path_fold, "rb") as f:
                fold_data = pickle.load(f)

            resized_fold = {}
            for key, value in fold_data.items():
                resized_sequences = []
                for i in range(len(value)):
                    seq = value[i]
                    seq = seq[50:151-50]
                    resized_sequences.append(seq)
                resized_fold[key] = resized_sequences


            n_negatives = len(resized_fold["negative"])
            quota = n_negatives // 4

            # (N,) boolean that is True exactly for the first occurrence of each 51-mer
            keep_first = np.zeros(len(all_seqs), dtype=bool)
            _, first_idx = np.unique(all_seqs, return_index=True)
            keep_first[first_idx] = True


            assert lower_bound < upper_bound, "Lower bound must be less than upper bound"
            if lower_bound < 0.5:
                bin_edges = [lower_bound, 0.5]
                edge = 0.5
                while edge < upper_bound:
                    edge = round(edge + 0.05, 2)
                    bin_edges.append(min(edge, upper_bound))
            else:
                bin_edges = [lower_bound]
                edge = lower_bound
                while edge < upper_bound:
                    edge = round(edge + 0.05, 2)
                    bin_edges.append(min(edge, upper_bound))

            # For every positive class build masks per bin
            class_bin_idx = {cls: [] for cls in (1, 2, 3, 4)}
            for cls in (1, 2, 3, 4):
                prob_vec = all_probs[:, cls]
                for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                    mask = (all_preds == cls) & (prob_vec >= lo) & (prob_vec < hi) & keep_first
                    idx  = np.where(mask)[0]
                    rng_fold.shuffle(idx)               # random order _within_ the bin
                    class_bin_idx[cls].append(list(idx))

            ########################################################################
            #  Round-robin picking
            ########################################################################
            for cls in (4, 3, 2, 1):                   # class priority unchanged
                selected  = []
                bins      = class_bin_idx[cls]
                num_bins  = len(bins)
                bin_ptr   = 0

                while len(selected) < quota:
                    # stop if _all_ bins are empty
                    if all(len(b) == 0 for b in bins):
                        break

                    # find next non-empty bin
                    start_ptr = bin_ptr
                    while len(bins[bin_ptr]) == 0:
                        bin_ptr = (bin_ptr + 1) % num_bins
                        if bin_ptr == start_ptr:        # wrapped around → all empty
                            break

                    if len(bins[bin_ptr]) == 0:
                        break                           # nothing left anywhere

                    i = bins[bin_ptr].pop()             # pop one index
                    seq = all_seqs[i]
                    if too_similar(seq, centres, seed_index):
                        continue                        # veto – try next round
                    selected.append(i)
                    idx_added_fold[cls].append(seq)

                    # register new centre so next iterations respect pigeon-hole
                    c21 = seq[15:36]
                    centres.append(c21)
                    cid = len(centres) - 1
                    seed_index[0][c21[:10]].append(cid)
                    seed_index[1][c21[11:]].append(cid)

                    bin_ptr = (bin_ptr + 1) % num_bins  # advance to next bin

                resized_fold["negative"].extend(all_seqs[selected].tolist())




            idx_added_dict[fold] = idx_added_fold
            path_fold_augmented = path_threshold / f"fold_{fold}_train.pickle"
            with open(path_fold_augmented, "wb") as f:
                pickle.dump(resized_fold, f)

        
        all_idx_added[(upper_bound, lower_bound)] = idx_added_dict


with open(folds_augmented / "idx_added_per_threshold_fold_class.pickle", "wb") as f:
    pickle.dump(all_idx_added, f)



