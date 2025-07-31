import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle


DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
import math
import itertools

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
        seq_len=51
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
        
        self.rng_init = random.Random(2000)
        self.rng_init.shuffle(self.negatives)
        self.negatives_epoch = len(self.class_1)*4
        
        if self.seq_len != len(self.class_1[0]):
                assert (len(self.class_1[0]) > self.seq_len) and (len(self.class_1[0]) % 2 != 0), \
                    "The sequence length must be odd and greater than the specified seq_len."
                cut_size = (len(self.class_1[0]) - self.seq_len) // 2
                self.class_1 = resize(self.class_1, cut_size)
                self.class_2 = resize(self.class_2, cut_size)
                self.class_3 = resize(self.class_3, cut_size)
                self.class_4 = resize(self.class_4, cut_size)
                self.negatives = resize(self.negatives, cut_size)

        self.set_epoch(0)


    def set_epoch(self, epoch: int):

        rng = random.Random(self.base_seed + epoch)
        # Calculate required negatives for the epoch (e.g., 4 * len(class_1))
        n_needed = self.negatives_epoch  # already set as 4 * len(class_1)
        # Determine how many negatives to shift (i.e., carry over) from the previous epoch.
        shift = 0.0
        shift_count = int(n_needed * shift)  # e.g., 50% of n_needed when shift=0.5

        n_total_neg = len(self.negatives)
        # Compute starting index using modulo arithmetic based on shift_count and epoch
        start = (shift_count * epoch) % n_total_neg

        if start + n_needed <= n_total_neg:
            negatives_now = self.negatives[start : start + n_needed]
        else:
            # Wrap around if we hit the end of negatives list
            negatives_now = self.negatives[start:] + self.negatives[: (start + n_needed - n_total_neg)]

        self.class_1 = sorted(self.class_1)
        self.class_2 = sorted(self.class_2)
        self.class_3 = sorted(self.class_3)
        self.class_4 = sorted(self.class_4)
        self.negatives_now = sorted(negatives_now)

        self.class_1 = rng.sample(self.class_1, len(self.class_1))
        self.class_2 = rng.sample(self.class_2, len(self.class_2))
        self.class_3 = rng.sample(self.class_3, len(self.class_3))
        self.class_4 = rng.sample(self.class_4, len(self.class_4))
        self.negatives_now = rng.sample(self.negatives_now, self.negatives_epoch)

        self.class_1_tokens = tokenize_sequences(self.class_1, embed=self.embed, kmer=self.kmer)
        self.class_2_tokens = tokenize_sequences(self.class_2,  embed=self.embed, kmer=self.kmer)
        self.class_3_tokens = tokenize_sequences(self.class_3,  embed=self.embed, kmer=self.kmer)
        self.class_4_tokens = tokenize_sequences(self.class_4,  embed=self.embed, kmer=self.kmer)
        self.negatives_tokens = tokenize_sequences(self.negatives_now, embed=self.embed, kmer=self.kmer)

        #one positive and one negative
        self.chunks = []
        self.chunk_labels = []

        
        #class 1 is the most represented so for the other classes we concatenate the other classes instances till we the same number of class 1
        class_2_multiplier = math.ceil(len(self.class_1_tokens) / len(self.class_2_tokens))
        class_3_multiplier = math.ceil(len(self.class_1_tokens) / len(self.class_3_tokens))
        class_4_multiplier = math.ceil(len(self.class_1_tokens) / len(self.class_4_tokens))
        #we shuffle every time we concatenate for the same class
        dummy_class_2 = self.class_2_tokens.copy()
        for i in range(class_2_multiplier):
            shuffled = rng.sample(dummy_class_2, len(dummy_class_2))
            self.class_2_tokens += shuffled
        dummy_class_3 = self.class_3_tokens.copy()
        for i in range(class_3_multiplier):
            shuffled = rng.sample(dummy_class_3, len(dummy_class_3))
            self.class_3_tokens += shuffled
        dummy_class_4 = self.class_4_tokens.copy()
        for i in range(class_4_multiplier):
            shuffled = rng.sample(dummy_class_4, len(dummy_class_4))
            self.class_4_tokens += shuffled

        for i in range(len(self.class_1_tokens)):
            self.chunk_labels.append(0)
            self.chunk_labels.append(0)
            self.chunk_labels.append(0)
            self.chunk_labels.append(0)
            self.chunks.extend(self.negatives_tokens[i*4:(i+1)*4])
            self.chunk_labels.append(1)
            self.chunks.append(self.class_1_tokens[i])
            self.chunk_labels.append(2)
            self.chunks.append(self.class_2_tokens[i])
            self.chunk_labels.append(3)
            self.chunks.append(self.class_3_tokens[i])
            self.chunk_labels.append(4)
            self.chunks.append(self.class_4_tokens[i])



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
        kmer=2,
        seq_len=51
    )

    
    # Iterate over the dataset
    for batch in dataset:
        input_ids = batch["input_ids"]
        label = batch["label"]
        print(input_ids, label)
        print(input_ids.shape, label.shape)
        #break  # just show one