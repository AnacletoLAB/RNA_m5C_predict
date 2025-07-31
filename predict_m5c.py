#!/usr/bin/env python3
"""
Predict RNA modifications from FASTA sequences with a hard‑negative‑mined Bi‑GRU.

Output file formats
-------------------
• **CSV** (default, comma‑separated) – give a filename ending in `.csv` or omit an extension.
• **TSV** (tab‑separated) – give a filename ending in `.tsv` or `.txt`.
• **Excel** – give a filename ending in `.xlsx` (requires the *openpyxl* package).

Columns in any format
---------------------
sequence_id | position | Type ("unmodified", "I", "II", "III", "IV") | p. unmodified | p. I | p. II | p. III | p. IV

Probabilities are rounded to 4 decimal places.

Examples
--------
```bash
# CSV (implicit)
python predict_rna_modifications.py --fasta_file sample.fasta

# Tab‑separated TSV
python predict_rna_modifications.py --fasta_file sample.fasta --output_file results.tsv

# Excel for biologists (requires `pip install openpyxl`)
python predict_rna_modifications.py --fasta_file sample.fasta --output_file results.xlsx

# Larger batch and CPU‑only
python predict_rna_modifications.py --fasta_file sample.fasta --batch_size 128 --cpu
```
"""

from experiments.models.RNNClassifier import RNNClassifier
from pathlib import Path
import torch
import argparse
import pandas as pd
from typing import List, Tuple, Dict
import sys

###############################################################################
# 1. Argument parsing
###############################################################################
parser = argparse.ArgumentParser(
    description="Predict RNA modifications from FASTA sequences.")
parser.add_argument("--fasta_file", type=str, default="test.fasta",
                    help="Path to the input FASTA file.")
parser.add_argument("--output_file", type=str, default="predictions.csv",
                    help="Output filename (extension decides format: .csv/.tsv/.xlsx).")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for prediction (default: 32).")
parser.add_argument("--cpu", action="store_true",
                    help="Run on CPU only, even if GPU is available.")
args = parser.parse_args()

###############################################################################
# 2. Device & model loading
###############################################################################
path_weights = Path(__file__).parent / "model_weights/heavy_hard_negative_mining_bigru.pt"
if not path_weights.exists():
    sys.exit(f"Weights not found: {path_weights}")

device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

params_model = {
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 3,
    "num_classes": 5,
    "rnn_type": "GRU",
    "bidirectional": True,
    "dropout": 0.1,
    "pooling": "central_attention",
    "seq_len": 51,
    "kmer": 1,
    "embed": "one_hot",
}

print("\nLoading model …", file=sys.stderr)
model = RNNClassifier(**params_model)
model.load_state_dict(torch.load(path_weights, map_location="cpu"))
model.to(device).eval()
print(f"Model loaded on {device}.", file=sys.stderr)

###############################################################################
# 3. FASTA parsing utilities
###############################################################################

def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header, seq_lines = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.upper())
        if header is not None:
            records.append((header, "".join(seq_lines)))
    if not records:
        sys.exit(f"No records found in FASTA {path}")
    return records

###############################################################################
# 4. Window generation around cytosines (C)
###############################################################################
WINDOW = params_model["seq_len"]
HALF = WINDOW // 2
ALLOWED = set("ACGT")
PADDING_CHAR = "P"


def iter_windows(seq: str):
    seq = seq.replace("U", "T")
    n = len(seq)
    for idx, base in enumerate(seq):
        if base != "C":
            continue
        start, end = idx - HALF, idx + HALF
        window_chars, invalid = [], False
        for pos in range(start, end + 1):
            if 0 <= pos < n:
                b = seq[pos]
                if b not in ALLOWED:
                    invalid = True
                    break
                window_chars.append(b)
            else:
                window_chars.append(PADDING_CHAR)
        if not invalid:
            yield "".join(window_chars), idx + 1

###############################################################################
# 5. One‑hot encoding
###############################################################################
ENCODING: Dict[str, List[int]] = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "P": [0, 0, 0, 0],
}


def encode_window(win: str) -> torch.Tensor:
    out = torch.empty(WINDOW, 4, dtype=torch.float32)
    for i, ch in enumerate(win):
        out[i] = torch.tensor(ENCODING[ch], dtype=torch.float32)
    return out

###############################################################################
# 6. Inference loop
###############################################################################
records = read_fasta(Path(args.fasta_file))
print(f"Loaded {len(records)} FASTA entries.", file=sys.stderr)

bsize = max(1, args.batch_size)
print(f"Batch size: {bsize}", file=sys.stderr)

TYPE_NAMES = ["unmodified", "I", "II", "III", "IV"]
prob_cols = [f"p. {n}" for n in TYPE_NAMES]

results: List[Dict[str, object]] = []
_batch, meta = [], []  # tensors and metadata


def flush_batch():
    if not _batch:
        return
    x = torch.stack(_batch).to(device)
    with torch.no_grad():
        ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.no_grad
        with ctx():
            logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_cls = probs.argmax(dim=1)
    for (seq_id, pos), row_prob, cls_idx in zip(meta, probs, pred_cls):
        row = {"sequence_id": seq_id, "position": pos, "Type": TYPE_NAMES[int(cls_idx)]}
        for label, p in zip(prob_cols, row_prob):
            val = float(p)
            row[label] = round(val, 4)
        results.append(row)
    _batch.clear(); meta.clear()

for hdr, seq in records:
    for win, pos in iter_windows(seq):
        _batch.append(encode_window(win)); meta.append((hdr, pos))
        if len(_batch) >= bsize:
            flush_batch()
flush_batch()

if not results:
    sys.exit("No valid cytosines found – nothing to predict.")

###############################################################################
# 7. Save predictions in chosen format
###############################################################################
output_path = Path(args.output_file)
if output_path.suffix == "":
    output_path = output_path.with_suffix(".csv")

print(f"Saving {len(results)} predictions to {output_path}", file=sys.stderr)

df = pd.DataFrame(results)
df = df[["sequence_id", "position", "Type"] + prob_cols]

ext = output_path.suffix.lower()
try:
    if ext in {".csv"}:
        df.to_csv(output_path, index=False)
    elif ext in {".tsv", ".txt"}:
        df.to_csv(output_path, sep="\t", index=False)
    elif ext in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False, engine="openpyxl")
    else:
        sys.exit(f"Unsupported output extension '{ext}'. Use .csv, .tsv, .txt, or .xlsx.")
except ImportError as e:
    sys.exit("Excel output requires the 'openpyxl' package. Run: pip install openpyxl")

print("Done.", file=sys.stderr)
