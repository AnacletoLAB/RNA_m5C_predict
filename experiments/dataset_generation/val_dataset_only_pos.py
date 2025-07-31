import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import math

if __name__ == "__main__":
    from tokenizer import tokenize_sequences
else:
    from dataset_generation.tokenizer import tokenize_sequences


def resize(list_seqs, cut_size):
    return [seq[cut_size:-cut_size] for seq in list_seqs]

class Datasetval(Dataset):
    """
    A dataset that:
    """
    def __init__(
        self,
        data_dict,
        base_seed=42,
        embed="one_hot",
        kmer=1,
        seq_len=51
    ):
        
        super().__init__()
        self.data_dict = data_dict
        self.base_seed = base_seed
        self.embed = embed
        self.kmer = kmer
        self.seq_len = seq_len


        self.class_1 = data_dict["type_1"]
        self.class_2 = data_dict["type_2"]
        self.class_3 = data_dict["type_3"]
        self.class_4 = data_dict["type_4"]

        if self.seq_len != len(self.class_1[0]):
                assert (len(self.class_1[0]) > self.seq_len) and (len(self.class_1[0]) % 2 != 0), \
                    "The sequence length must be odd and greater than the specified seq_len."
                cut_size = (len(self.class_1[0]) - self.seq_len) // 2
                self.class_1 = resize(self.class_1, cut_size)
                self.class_2 = resize(self.class_2, cut_size)
                self.class_3 = resize(self.class_3, cut_size)
                self.class_4 = resize(self.class_4, cut_size)

        self.set_epoch(0)


    def set_epoch(self, epoch: int):


        self.class_1_tokens = tokenize_sequences(self.class_1, embed=self.embed, kmer=self.kmer)
        self.class_2_tokens = tokenize_sequences(self.class_2,  embed=self.embed, kmer=self.kmer)
        self.class_3_tokens = tokenize_sequences(self.class_3,  embed=self.embed, kmer=self.kmer)
        self.class_4_tokens = tokenize_sequences(self.class_4,  embed=self.embed, kmer=self.kmer)


    
        self.chunks = []
        self.chunks.extend(self.class_1_tokens)
        self.chunks.extend(self.class_2_tokens)
        self.chunks.extend(self.class_3_tokens)
        self.chunks.extend(self.class_4_tokens)

        self.chunk_labels = []
        self.chunk_labels.extend([1]*len(self.class_1_tokens))
        self.chunk_labels.extend([2]*len(self.class_2_tokens))
        self.chunk_labels.extend([3]*len(self.class_3_tokens))
        self.chunk_labels.extend([4]*len(self.class_4_tokens))

        

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
