import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from pathlib import Path
import sys, pickle, json
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm
import argparse

#insert path script of the weights if custom
parser = argparse.ArgumentParser(description="Inference script for transcriptome models")
parser.add_argument("--path_weights", type=str, default=None,
                    help="Path to the model weights file. If not provided, will use default paths.")
parser.add_argument("--model_name", type=str, default="RNN",
                    choices=["RNN", "Transformer", "1DCNN"],
                    help="Name of the model to use for inference.")
args = parser.parse_args()



SCRIPT_DIR = Path(__file__).parent
WORKING_DIR = SCRIPT_DIR.parent

sys.path.append(str(WORKING_DIR))

mapping_models = {
    "RNN": {"path_conf":WORKING_DIR / "models_params/RNN_config.json",
            "path_weights":WORKING_DIR / "logs/RNN/RNN_epoch_43.pt"},
    "Transformer": {"path_conf":WORKING_DIR / "models_params/Transformer_config.json",
                    "path_weights": WORKING_DIR / "logs/Transformer/Transformer_epoch_23.pt"},
    "1DCNN": {"path_conf":WORKING_DIR / "models_params/1DCNN_config.json",
                    "path_weights": WORKING_DIR / "logs/1DCNN/1DCNN_epoch_50.pt"},
}
from utils.search_space import model_mapping

model_name = args.model_name
if args.path_weights:
    data_path = Path(args.path_weights)
else:
    data_path = mapping_models[model_name]["path_weights"]

print(f"Using model: {data_path}")

class CytWindowDataset(Dataset):
    """
    One item  = one 51-mer centred on a C, plus its transcript index & position.
    """

    def __init__(self, npz_path):
        super().__init__()

        self.one_hot_lut = torch.tensor([
            [0,0,0,0],
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ], dtype=torch.float32)

        data         = np.load(npz_path, allow_pickle=True)
        self.id2tx   = data["id2tx"]            # (N_transcripts,)
        self.seqs    = data["seq_arrs"]         # list of uint8 arrays
        self.c_pos   = data["c_pos_arrs"]       # list of uint32 arrays

        # Flatten (tx_idx, pos) into two big vectors for O(1) indexing
        tx_idx, pos  = [], []
        for i, arr in enumerate(self.c_pos):
            tx_idx.extend([i] * len(arr))
            pos.extend(arr)

        self.tx_idx = np.asarray(tx_idx, dtype=np.uint32)
        self.pos    = np.asarray(pos,    dtype=np.uint32)


        #print len of the dataset to see how many items it has
        print(f"Dataset size: {len(self.tx_idx)} items")
        #number of batches
        print(f"Number of batches: {len(self.tx_idx) // 2048}")

    def __len__(self):
        return len(self.tx_idx)

    def __getitem__(self, idx):


        tx  = self.tx_idx[idx]
        p   = self.pos[idx]
        win = self.seqs[tx][p-25:p+26]               # uint8 array
        one_hot = self.one_hot_lut[torch.from_numpy(win).long()]

        return {
            "window"     : one_hot,
            "tx_idx"     : int(tx),
            "center_pos" : int(p)
        }




path_config = mapping_models[model_name]["path_conf"]


    

with open(path_config, 'r') as f:
    config = json.load(f)

model_params = config["model_params"]

model = model_mapping(model_name)(**model_params)
model.load_state_dict(torch.load(data_path, map_location=torch.device('cpu')))
model.eval()
assert torch.cuda.is_available(), "Inference requires a GPU"
model = model.cuda()
print("Using GPU")


from torch.utils.data import DataLoader

ds = CytWindowDataset(SCRIPT_DIR / ("transcriptome/tx_data.npz"))

batch_size = 65536

loader = DataLoader(
    ds,
    batch_size   = batch_size,          # tune to half your VRAM  batch = 65536
    num_workers  = 8,
    pin_memory   = True,
    persistent_workers = True,
)




schema = pa.schema([
    ("tx_idx",      pa.uint32()),
    ("center_pos",  pa.uint32()),
    ("prob0",       pa.float16()),
    ("prob1",       pa.float16()),
    ("prob2",       pa.float16()),
    ("prob3",       pa.float16()),
    ("prob4",       pa.float16()),
])

if not args.path_weights:
    writer = pq.ParquetWriter(SCRIPT_DIR / (f"transcriptome/{model_name}_predictions.parquet"), schema)
else:
    # to_save = data_path.stem.split(".")[0]
    # writer = pq.ParquetWriter(SCRIPT_DIR / (f"transcriptome/{to_save}_predictions.parquet"), schema)
    to_save = str(str(data_path).split("/")[-3:]).strip()
    writer = pq.ParquetWriter(SCRIPT_DIR / (f"transcriptome/{to_save}_predictions.parquet"), schema)

total_batches = len(loader)

for i, batch in enumerate(tqdm(loader, desc=f"Inferencing {model_name}", total=total_batches)):
    with torch.no_grad(), torch.cuda.amp.autocast():
        x   = batch["window"].to("cuda").float()
        out = model(x)                                # [B, 5] logits
        out = F.softmax(out, dim=1)                  # [B, 5] probabilities


    tbl = pa.Table.from_pydict({
        "tx_idx"     : batch["tx_idx"],
        "center_pos" : batch["center_pos"],
        "prob0"      : out[:,0].half().cpu().numpy(),
        "prob1"      : out[:,1].half().cpu().numpy(),
        "prob2"      : out[:,2].half().cpu().numpy(),
        "prob3"      : out[:,3].half().cpu().numpy(),
        "prob4"      : out[:,4].half().cpu().numpy(),
    }, schema=schema)
    writer.write_table(tbl)

writer.close()