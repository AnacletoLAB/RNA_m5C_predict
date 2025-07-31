import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import math
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from tokenizer import tokenize_sequences
else:
    from dataset_generation.tokenizer import tokenize_sequences

def resize(list_seqs, cut_size):
    return [seq[cut_size:-cut_size] for seq in list_seqs]

class Dataset_(Dataset):
    """
    A dataset that:
    """
    def __init__(
        self,
        df_,
        base_seed=42,
        embed="one_hot",
        kmer=1,
        seq_len=51
    ):
        
        super().__init__()
        self.df_ = df_
        self.base_seed = base_seed
        self.embed = embed
        self.kmer = kmer
        self.seq_len = seq_len
        
        self.seqs_train = df_.iloc[:, 0].tolist()
        self.seqs_train = ["".join([y if y != "U" else "T" for y in x]) for x in self.seqs_train]
        self.seqs_train = ["".join([y if y != "N" else "P" for y in x]) for x in self.seqs_train]
        self.chunk_labels = df_.iloc[:, 1].tolist()

        if self.seq_len != len(self.seqs_train[0]):
            assert (len(self.seqs_train[0]) > self.seq_len) and (len(self.seqs_train[0]) % 2 != 0), \
                "The sequence length must be odd and greater than the specified seq_len."
            cut_size = (len(self.seqs_train[0]) - self.seq_len) // 2
            self.seqs_train = resize(self.seqs_train, cut_size)

        self.chunks = tokenize_sequences(self.seqs_train, embed=self.embed, kmer=self.kmer)
    


    def set_epoch(self, epoch: int):

        rng = random.Random(self.base_seed + epoch)

        combined = list(zip(self.chunks, self.chunk_labels))
        rng.shuffle(combined)
        self.chunks, self.chunk_labels = zip(*combined)
        self.chunks = list(self.chunks)
        self.chunk_labels = list(self.chunk_labels)



    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        Return tokens & sub-sampled labels for chunk #idx
        """

        chunk_tokens = self.chunks[idx]
        chunk_labels = self.chunk_labels[idx]

        if self.embed == "embeddings":
            tokens_tensor = torch.tensor(chunk_tokens, dtype=torch.long)
        else:
            tokens_tensor = torch.tensor(chunk_tokens, dtype=torch.float)

        label_tensor = torch.tensor(chunk_labels, dtype=torch.long)

        return {
            "input_ids": tokens_tensor,
            "label": label_tensor,
        }


if __name__ == "__main__":

    path_train = Path.cwd().parent / "data_mlm5c/train_m5C_201_2.txt"
    
    df_train = pd.read_csv(path_train, sep=",", header=None)
    
    # Create the dataset
    dataset = Dataset_(
        df_=df_train,
        base_seed=42,
        embed="one_hot",
        kmer=1,
        seq_len=41
    )
    dataset.set_epoch(0)
    # Iterate over the dataset
    for batch in dataset:
        input_ids = batch["input_ids"]
        label = batch["label"]
        print(input_ids, label)
        print(input_ids.shape, label.shape)
        #break  # just show one