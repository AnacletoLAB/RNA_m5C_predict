# RNA m⁵C Predictor

> Deep‑learning inference for cytosine 5‑methylation in Human RNA sequences

![Pipeline overview](overview.pdf)
End‑to‑end pipeline for writer‑resolved m5C prediction.
**(1) Reconstruction of an m5C high-confidence catalog**. The 45\,500 bisulfite calls} are lifted over, mapped to GENCODE v19 and cytosines on methylated transcripts are split into methylated (turquoise) and unmethylated (orange). Nine STREME runs plus manual curation yield four writer motifs (Type I–IV, boxed). PFM‑based rescoring removes BS‑seq artifacts, random negatives are added from the same transcripts and redundancy filtering produces a 26\,300×2 non‑redundant corpus. **(2) Model Training and Inference**. A five‑fold CV grid search (Bi‑GRU, CNN, Transformer) selects Bi‑GRU as the best model. Its false‑positive calls on around 10 million held‑out cytosines are harvested as hard negatives, redundancy-filtered and merged into an augmented training set, then re‑trained and deployed transcriptome‑wide. **(3) Resources**. The resulting writer‑specific prediction database, with refined motifs and coherent secondary‑structure profiles are released for community use. In this repository we further provide a standalone Python tool for m\textsuperscript{5}C prediction given FASTA files as described below.

This repository accompanies **Saitto *********************************et al.********************************* 2025** (manuscript in prep.)  &#x20;

---

## What’s inside

```
📂 dataset/         # training/validation data
📂 experiments/     # training + analysis code
📂 human_transcriptome_predictions/     # dataframe with predicted m5Cs across human trascriptome
📂 model_weights/   # final Bi‑GRU checkpoint (heavy hard‑negative mining)
predict_m5c.py      # ← run this to predict new samples
test.fasta          # tiny example FASTA for a smoke test
```

---

## How `predict_m5c.py` works

1. **Loads** the frozen Bi‑GRU from `model_weights/…pt`.
2. **Parses** an input FASTA (DNA or RNA; `U` is transparently mapped to `T`).
3. **Slides** a 51‑nt window over every cytosine (25 nt flanks).  Windows
   containing ambiguous bases (`N`, etc.) are skipped.
4. **One‑hot encodes** each window to `(51, 4)` and batches them.
5. **Predicts** five classes *(unmodified, I, II, III, IV)* and writes a
   probability table.

---

## Requirements (inference)

| Package     | Version used |
| ----------- | ------------ |
| **Python**  |  3.11.5      |
| **PyTorch** |  2.1.0       |
| **pandas**  |  2.2.3       |

Optional for Excel output: **openpyxl ≥ 3.1.0**

> The script auto‑detects a GPU. Pass `--cpu` to force CPU‑only inference
> (works even with a CPU‑only PyTorch wheel).

---

## Installation

```bash
python -m pip install --upgrade pip
pip install torch==2.1.0 pandas==2.2.3
# Excel support (optional)
pip install "openpyxl>=3.1.0"
```

---

## Usage

```bash
#place ".fasta" file inside repository

# quick start (GPU if available)
python predict_m5c.py --fasta_file my_sequences.fasta

# force CPU
python predict_m5c.py --fasta_file my_sequences.fasta --cpu

# change batch size (faster on large GPUs)
python predict_m5c.py --fasta_file my_sequences.fasta --batch_size 256

# choose output format by extension
python predict_m5c.py --fasta_file my_sequences.fasta --output_file results.tsv   # TSV
python predict_m5c.py --fasta_file my_sequences.fasta --output_file results.xlsx  # Excel (needs openpyxl)
```

### Output columns

```
sequence_id | position | Type (unmodified/I/II/III/IV) | p. unmodified | p. I | p. II | p. III | p. IV
```

---

## Data availability

| Location                         | File                                                      | Size  | Purpose                                                      |
| -------------------------------- | --------------------------------------------------------- | ----- | ------------------------------------------------------------ |
| **GitHub** (`human_transcriptome_predictions/`) | `m5C_predictions.tsv.gz`                                  | 4 MB  | Quick download; small enough for the repo.                   |
| **Zenodo**                       | `m5C_predictions.tsv.gz` <br> *(same checksum as GitHub)* | 4 MB  | Archival copy with DOI for citation; long‑term preservation. |
| *(optional)* Zenodo              | `m5C_predictions.xlsx`                                    | 16 MB | Excel version for bench biologists (exported from the TSV).  |

### Dataset description

This dataset provides **transcriptome‑wide predictions of RNA 5‑methyl‑cytosine (m⁵C) sites** for the human reference transcriptome (GENCODE v45, GRCh38). Predictions were generated with the Bi‑GRU model described in **Saitto *et al.* 2025**.

Each row corresponds to a cytosine residue predicted to be methylated and contains:

* **Transcript‑level identifiers:** `transcript_id`, `gene_id`, `gene_name`, `transcript_type`, `tags`.
* **`position`:** zero‑based coordinate of the cytosine within the transcript sequence.
* **`Type`:** predicted methyltransferase class – I (NSUN2), II (NSUN6), III (NSUN5), IV (NSUN1).
* **`probability`:** posterior probability assigned by the model (rounded to 4 decimals).
* **`in_train_or_test_sets`:** `TRUE` if the 51‑nt window centred on this cytosine was present in the training *or* validation sets; `FALSE` otherwise.

File formats:

* **`m5C_predictions.tsv.gz`** – tab‑separated, UTF‑8, gzip‑compressed (4 MB).
* **`m5C_predictions.xlsx`** – Excel workbook with identical content (25 MB).



```text
Data DOI: https://doi.org/10.5281/zenodo.16629378
```

---

## Citation

> Saitto *et al.* **Hard negative mining reveals the decisive role of data quality over model complexity in RNA m5C prediction.** 2025. DOI: **10.XXXX/placeholder‑doi**


---
