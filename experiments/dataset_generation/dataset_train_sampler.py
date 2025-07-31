import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
import math
import numpy as np

if __name__ == "__main__":
    from tokenizer import tokenize_sequences
else:
    from dataset_generation.tokenizer import tokenize_sequences

def resize(list_seqs, cut_size):
    return [seq[cut_size:-cut_size] for seq in list_seqs]

class Datasettrain(Dataset):
    """
    A dataset that:
    """
    def __init__(
        self,
        data_dict,
        base_seed=42,
        embed="one_hot",
        kmer=1,
        seq_len=51,
        same_neg_num = True
    ):
        
        super().__init__()
        self.data_dict = data_dict
        self.base_seed = base_seed
        self.embed = embed
        self.kmer = kmer
        self.seq_len = seq_len
        
        #four classes
        self.negatives =  data_dict["negative"]
        self.class_1 = data_dict["type_1"]
        self.class_2 = data_dict["type_2"]
        self.class_3 = data_dict["type_3"]
        self.class_4 = data_dict["type_4"]
        
        if same_neg_num:
            len_negatives = len(self.class_1) + len(self.class_2) + len(self.class_3) + len(self.class_4)
            self.negatives = self.negatives[:len_negatives]

        
        if self.seq_len != len(self.class_1[0]):
                assert (len(self.class_1[0]) > self.seq_len) and (len(self.class_1[0]) % 2 != 0), \
                    "The sequence length must be odd and greater than the specified seq_len."
                cut_size = (len(self.class_1[0]) - self.seq_len) // 2
                self.class_1 = resize(self.class_1, cut_size)
                self.class_2 = resize(self.class_2, cut_size)
                self.class_3 = resize(self.class_3, cut_size)
                self.class_4 = resize(self.class_4, cut_size)
                self.negatives = resize(self.negatives, cut_size)

        self.class_1_tokens = tokenize_sequences(self.class_1, embed=self.embed, kmer=self.kmer)
        self.class_2_tokens = tokenize_sequences(self.class_2,  embed=self.embed, kmer=self.kmer)
        self.class_3_tokens = tokenize_sequences(self.class_3,  embed=self.embed, kmer=self.kmer)
        self.class_4_tokens = tokenize_sequences(self.class_4,  embed=self.embed, kmer=self.kmer)
        self.negatives_tokens = tokenize_sequences(self.negatives, embed=self.embed, kmer=self.kmer)


        #one positive and one negative
        self.chunks = []
        self.chunks.extend(self.negatives_tokens)
        self.chunks.extend(self.class_1_tokens)
        self.chunks.extend(self.class_2_tokens)
        self.chunks.extend(self.class_3_tokens)
        self.chunks.extend(self.class_4_tokens)

        self.chunk_labels = []
        self.chunk_labels.extend([0]*len(self.negatives_tokens))
        self.chunk_labels.extend([1]*len(self.class_1_tokens))
        self.chunk_labels.extend([2]*len(self.class_2_tokens))
        self.chunk_labels.extend([3]*len(self.class_3_tokens))
        self.chunk_labels.extend([4]*len(self.class_4_tokens))
        

        self.set_weights()


    # def set_epoch(self, epoch: int):

    #     rng = random.Random(self.base_seed + epoch)

    #     combined = list(zip(self.chunks, self.chunk_labels))
    #     rng.shuffle(combined)
    #     # unzip back to separate lists
    #     self.chunks, self.chunk_labels = zip(*combined)
    #     # convert back to lists
    #     self.chunks = list(self.chunks)
    #     self.chunk_labels = list(self.chunk_labels)

    def set_weights(self, pi=None):
        labels = np.asarray(self.chunk_labels)
        counts = np.bincount(labels)
        n_neg, n_pos_total = counts[0], counts[1:].sum()
        if pi is None:
            pi = np.zeros_like(counts, dtype=float)
            pi[0]      = 0.5
            pi[1:]     = 0.5 / 4
        else:
            pi = np.asarray(pi, dtype=float)
            if len(pi) != len(counts):
                raise ValueError(f"pi must have the same length as counts. Got {len(pi)} vs {len(counts)}")
            if not np.isclose(pi.sum(), 1.0):
                raise ValueError(f"pi must sum to 1.0. Got {pi.sum()}")
            
        self.weights = pi[labels] / counts[labels]
        self.weights = torch.tensor(self.weights, dtype=torch.double)


        
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
            "label": label_tensor
        }


if __name__ == "__main__":

    path_train = DATASET_DIR / "full_dataset_final_training.pickle"
    
    with open(path_train, 'rb') as f:
        train_dict = pickle.load(f)
    
    # Create the dataset
    dataset = Datasettrain(
        data_dict=train_dict,
        base_seed=42,
        embed="one_hot",
        kmer=1,
        seq_len=41
    )

    dataset.set_weights()
    
    # Iterate over the dataset
    for batch in dataset:
        input_ids = batch["input_ids"]
        label = batch["label"]
        print(input_ids, label)
        print(input_ids.shape, label.shape)
        #break  # just show one