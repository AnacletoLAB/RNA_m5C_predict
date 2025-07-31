from pathlib import Path
import pandas as pd
import numpy as np
import pickle

def generate_data(which_data="mlm5c", seed=123):
    """
    Generates data for the specified dataset.
    """
    np.random.seed(seed)
    if which_data == "mlm5c":
        path_train = Path("./data_mlm5c/train_m5C_201_2.txt")
        path_test  = Path("./data_mlm5c/test_m5C_201_2.txt")
        df_train_all = pd.read_csv(path_train, sep=",", header=None)
        df_test  = pd.read_csv(path_test,  sep=",", header=None)

    elif which_data == "deepm5C":

        path_train = Path("./deepm5C_DataSet/train-5mC-mm.txt")
        path_test  = Path("./deepm5C_DataSet/ind-5mC-mm.txt")
        df_train_all = pd.read_csv(path_train, sep="\t", header=None, skiprows=1)
        df_test  = pd.read_csv(path_test,  sep="\t", header=None, skiprows=1)
        df_train_all[1] = df_train_all[1].apply(lambda x: x if x == 1 else 0)
        df_test[1] = df_test[1].apply(lambda x: x if x == 1 else 0)

    elif which_data == "current_dataset":
        path_dataset = Path(__file__).parents[2] / "dataset"
        path_train = path_dataset / "folds/fold_1_train.pickle"
        path_val   = path_dataset / "folds/fold_1_val.pickle"
        path_test  = path_dataset / "test_set.pickle"
        with open(path_train, "rb") as f:
            train_data = pickle.load(f)
        with open(path_val, "rb") as f:
            val_data = pickle.load(f)
        with open(path_test, "rb") as f:
            test_data = pickle.load(f)

        positive_keys = ['type_1', 'type_2', 'type_3', 'type_4']
        negative_keys = ["negative"]
        positive_seqs_train = [seq for key in positive_keys for seq in train_data[key]]
        negative_seqs_train = [seq for key in negative_keys for seq in train_data[key]]
        positive_seqs_val = [seq for key in positive_keys for seq in val_data[key]]
        negative_seqs_val = [seq for key in negative_keys for seq in val_data[key]]
        positive_seqs_test = [seq for key in positive_keys for seq in test_data[key]]
        negative_seqs_test = [seq for key in negative_keys for seq in test_data[key]]

        train_y = [1] * len(positive_seqs_train) + [0] * len(negative_seqs_train)
        val_y = np.array([1] * len(positive_seqs_val) + [0] * len(negative_seqs_val), dtype=int)
        test_y = np.array([1] * len(positive_seqs_test) + [0] * len(negative_seqs_test), dtype=int)

        #make dataframe with colum 0 as sequences and column 1 as labels
        df_train = pd.DataFrame({
            0: positive_seqs_train + negative_seqs_train,
            1: train_y
        })
        df_val = pd.DataFrame({
            0: positive_seqs_val + negative_seqs_val,
            1: val_y
        })
        df_test = pd.DataFrame({
            0: positive_seqs_test + negative_seqs_test,
            1: test_y
        })

    else:
        raise ValueError(f"Unsupported dataset: {which_data}")

    if which_data != "current_dataset":
        positives_train_indices = df_train_all[df_train_all[1] == 1].index.to_list()
        negatives_train_indices = df_train_all[df_train_all[1] == 0].index.to_list()

        np.random.shuffle(positives_train_indices)
        np.random.shuffle(negatives_train_indices)

        train_indices = positives_train_indices[:int(len(positives_train_indices)*0.8)] + negatives_train_indices[:int(len(negatives_train_indices)*0.8)]
        val_indices = positives_train_indices[int(len(positives_train_indices)*0.8):] + negatives_train_indices[int(len(negatives_train_indices)*0.8):]

        df_train = df_train_all.iloc[train_indices, :].reset_index(drop=True)
        df_val = df_train_all.iloc[val_indices, :].reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)


    return df_train, df_val, df_test