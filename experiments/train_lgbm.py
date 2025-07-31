import itertools
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

np.random.seed(123)




dataset_name = "mlm5c"  # "mlm5c" or "current_dataset" # or "negatives_vs_negatives"
val_thr = False

def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        if "P" not in sequence[i:i + k]:
            kmer.append(sequence[i:i + k])
    return kmer


def Kmer(sequences, k=3, NA = 'ACGU', normalize=True):

    encodings = []

    header = []
    for kmer in itertools.product(NA, repeat=k):
        header.append(''.join(kmer))

    for sequence in sequences:

        kmers = kmerArray(sequence, k)

        count = Counter()
        count.update(kmers)

        if normalize == True:
            for key in count:
                count[key] = count[key] / len(kmers)

        code = []
        for j in range(0, len(header)):
            if header[j] in count:
                code.append(count[header[j]])
            else:
                code.append(0)
    
        encodings.append(code)

    return encodings


if dataset_name == "mlm5c":

    path_train = Path("./data_mlm5c/train_m5C_201_2.txt")
    path_test  = Path("./data_mlm5c/test_m5C_201_2.txt")

    df_train = pd.read_csv(path_train, sep=",", header=None)
    df_test  = pd.read_csv(path_test,  sep=",", header=None)


    positives_train_indices = df_train[df_train[1] == 1].index.to_list()
    negatives_train_indices = df_train[df_train[1] == 0].index.to_list()

    np.random.shuffle(positives_train_indices)
    np.random.shuffle(negatives_train_indices)

    train_indices = positives_train_indices[:int(len(positives_train_indices)*0.8)] + negatives_train_indices[:int(len(negatives_train_indices)*0.8)]
    val_indices = positives_train_indices[int(len(positives_train_indices)*0.8):] + negatives_train_indices[int(len(negatives_train_indices)*0.8):]


    seqs_train = df_train.iloc[train_indices, 0].tolist()
    train_y = np.array(df_train.iloc[train_indices, 1].tolist(), dtype=int)

    seqs_val = df_train.iloc[val_indices, 0].tolist()
    val_y = np.array(df_train.iloc[val_indices, 1].tolist(), dtype=int)

    seqs_test = df_test.iloc[:, 0].tolist()
    test_y = np.array(df_test.iloc[:, 1].tolist(), dtype=int)



elif dataset_name == "current_dataset":

    DATASET_DIR = Path(__file__).parent.parent / "dataset"

    path_train = DATASET_DIR / "folds/fold_1_train.pickle"
    path_val  = DATASET_DIR / "folds/fold_1_val.pickle"
    path_test  = DATASET_DIR / "test_set.pickle"


    with open(path_train, "rb") as f:
        train_data = pickle.load(f)
    with open(path_val, "rb") as f:
        val_data = pickle.load(f)
    with open(path_test, "rb") as f:
        test_data = pickle.load(f)


    positive_keys = ['type_1', 'type_2', 'type_3', 'type_4']
    negative_keys = ["negative"]
    positive_seqs_train = ["".join([x if x != "T" else "U" for x in seq]) for key in positive_keys for seq in train_data[key]]
    negative_seqs_train = ["".join([x if x != "T" else "U" for x in seq]) for key in negative_keys for seq in train_data[key]]
    positive_seqs_val = ["".join([x if x != "T" else "U" for x in seq]) for key in positive_keys for seq in val_data[key]]
    negative_seqs_val = ["".join([x if x != "T" else "U" for x in seq]) for key in negative_keys for seq in val_data[key]]
    positive_seqs_test = ["".join([x if x != "T" else "U" for x in seq]) for key in positive_keys for seq in test_data[key]]
    negative_seqs_test = ["".join([x if x != "T" else "U" for x in seq]) for key in negative_keys for seq in test_data[key]]

    train_y = [1] * len(positive_seqs_train) + [0] * len(negative_seqs_train)
    val_y = np.array([1] * len(positive_seqs_val) + [0] * len(negative_seqs_val), dtype=int)
    test_y = np.array([1] * len(positive_seqs_test) + [0] * len(negative_seqs_test), dtype=int)

    seqs_train = positive_seqs_train + negative_seqs_train
    seqs_val = positive_seqs_val + negative_seqs_val
    seqs_test = positive_seqs_test + negative_seqs_test

    combined = list(zip(seqs_train, train_y))
    np.random.shuffle(combined)
    seqs_train, train_y = zip(*combined)

    train_y = np.array(train_y, dtype=int)

elif dataset_name == "negatives_vs_negatives":
    
    DATASET_DIR = Path(__file__).parent.parent / "dataset"

    path_train = DATASET_DIR / "folds/fold_1_train.pickle"
    path_val  = DATASET_DIR / "folds/fold_1_val.pickle"
    path_test  = DATASET_DIR / "test_set.pickle"


    with open(path_train, "rb") as f:
        train_data = pickle.load(f)
    with open(path_val, "rb") as f:
        val_data = pickle.load(f)
    with open(path_test, "rb") as f:
        test_data = pickle.load(f)

    negative_seqs_train = ["".join([x if x != "T" else "U" for x in seq]) for seq in train_data["negative"]]
    negative_seqs_val = ["".join([x if x != "T" else "U" for x in seq]) for seq in val_data["negative"]]
    negative_seqs_test = ["".join([x if x != "T" else "U" for x in seq]) for seq in test_data["negative"]]

    negatives_current = negative_seqs_train + negative_seqs_val + negative_seqs_test

    path_train = Path("./data_mlm5c/train_m5C_201_2.txt")
    path_test  = Path("./data_mlm5c/test_m5C_201_2.txt")

    df_train = pd.read_csv(path_train, sep=",", header=None)
    df_test  = pd.read_csv(path_test,  sep=",", header=None)

    negatives_mlm5c_train = df_train[df_train[1] == 0][0].to_list()
    negatives_mlm5c_test = df_test[df_test[1] == 0][0].to_list()
    negatives_mlm5c = negatives_mlm5c_train + negatives_mlm5c_test

    #resize negatives_mlm5c to match the length of negatives_current
    negatives_mlm5c = [x[25:-25] for x in negatives_mlm5c]

    #15% for test, the remaining 80% for training and 20% for validation
    np.random.shuffle(negatives_current)
    negatives_current = negatives_current[:len(negatives_mlm5c)]
    test_seqs_current = negatives_current[:int(len(negatives_current) * 0.15)]
    train_val_seqs_current = negatives_current[int(len(negatives_current) * 0.15):]
    train_seqs_current = train_val_seqs_current[:int(len(train_val_seqs_current) * 0.8)]
    val_seqs_current = train_val_seqs_current[int(len(train_val_seqs_current) * 0.8):]

    np.random.shuffle(negatives_mlm5c)
    test_seqs_mlm5c = negatives_mlm5c[:int(len(negatives_mlm5c) * 0.15)]
    train_val_seqs_mlm5c = negatives_mlm5c[int(len(negatives_mlm5c) * 0.15):]
    train_seqs_mlm5c = train_val_seqs_mlm5c[:int(len(train_val_seqs_mlm5c) * 0.8)]
    val_seqs_mlm5c = train_val_seqs_mlm5c[int(len(train_val_seqs_mlm5c) * 0.8):]

    seqs_train = train_seqs_current + train_seqs_mlm5c
    train_y = [1] * len(train_seqs_current) + [0] * len(train_seqs_mlm5c)

    seqs_val = val_seqs_current + val_seqs_mlm5c
    val_y = np.array([1] * len(val_seqs_current) + [0] * len(val_seqs_mlm5c))

    seqs_test = test_seqs_current + test_seqs_mlm5c
    test_y = np.array([1] * len(test_seqs_current) + [0] * len(test_seqs_mlm5c))

    combined = list(zip(seqs_train, train_y))
    np.random.shuffle(combined)
    seqs_train, train_y = zip(*combined)
    train_y = np.array(train_y, dtype=int)




else:
    raise ValueError("Unknown dataset name. Please choose 'mlm5c' or 'current_dataset'.")


train_X = np.array(Kmer(seqs_train), dtype=float)
val_X  = np.array(Kmer(seqs_val), dtype=float)
test_X  = np.array(Kmer(seqs_test), dtype=float)

lgb_train = lgb.Dataset(train_X, label=train_y)
lgb_eval  = lgb.Dataset(val_X,  label=val_y, reference=lgb_train)

params = {
    'objective': 'binary',    # binary classification
    'metric':    'auc',       # watch AUC on the validation set
    'verbosity': -1,          # no logs except eval
    'random_state': 123,      # reproducibility
}


clf = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_eval],          # use the val set for early stopping
    verbose_eval=50,                # print metrics every 50 iterations
    num_boost_round=1000,           # at most 1,000 boosting rounds
    early_stopping_rounds=100       # stop if no AUC‐gain for 100 rounds
)

val_pred_proba  = clf.predict(val_X,  num_iteration=clf.best_iteration)
test_pred_proba = clf.predict(test_X, num_iteration=clf.best_iteration)

if val_thr:
    prec, rec, thresh = precision_recall_curve(val_y, val_pred_proba)
    f1_vals = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.nanargmax(f1_vals)
    best_thresh = thresh[best_idx]

    test_pred = (test_pred_proba >= best_thresh).astype(int)
else:
    test_pred = (test_pred_proba >= 0.5).astype(int)


auc_test  = roc_auc_score(test_y, test_pred_proba)
acc_test  = accuracy_score(test_y, test_pred)
rec_test  = recall_score(test_y, test_pred)
prec_test = precision_score(test_y, test_pred)
f1_test   = f1_score(test_y, test_pred)
auprc_test = average_precision_score(test_y, test_pred_proba)

print(f"Test AUC: {auc_test:.4f}")
print(f"Test Accuracy: {acc_test:.4f}")
print(f"Test Recall: {rec_test:.4f}")
print(f"Test Precision: {prec_test:.4f}")
print(f"Test F1: {f1_test:.4f}")
print(f"Test AUPRC: {auprc_test:.4f}")



results_dir = Path("./results_lgb")
results_dir.mkdir(parents=True, exist_ok=True)


results_df = pd.DataFrame({
    "auc": [auc_test],
    "accuracy": [acc_test],
    "recall": [rec_test],
    "precision": [prec_test],
    "f1": [f1_test],
    "auprc": [auprc_test],
})

if val_thr:
    results_df.to_csv(results_dir / f"results_{dataset_name}_thr.csv", index=False)
    clf.save_model(str(results_dir / f"{dataset_name}_lgbm_thr.txt"))

else:
    results_df.to_csv(results_dir / f"results_{dataset_name}.csv", index=False)
    clf.save_model(str(results_dir / f"{dataset_name}_lgbm.txt"))

