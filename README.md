# Enhancing Taglish Sentiment Classification with calamanCy + Bi-LSTM

Thesis project comparing a FastText-only baseline Bi-LSTM against a proposed
calamanCy-enhanced Bi-LSTM for Taglish (Tagalog-English) code-switched sentiment
classification, benchmarked against Cosme & De Leon (2024) on the FiReCS dataset.

## The two models

| | Input to the Bi-LSTM, per token |
|---|---|
| **Baseline** | FastText vector (300) |
| **Proposed** | FastText vector (300) + calamanCy POS & dependency tags, fused (332) |

Both use the same Bi-LSTM, the same hyperparameters and the same tokens, so the
only difference is the calamanCy features.

**Current proposed design:** calamanCy `tl_calamancy_md-0.2.0` tags, **projection**
fusion: the one-hot POS + dependency tags pass through a small learned layer
(52 → 32), the FastText vector and the tag vector each get their own LayerNorm,
then they are concatenated. The original manuscript design (one LayerNorm over the
raw concatenation) is still available as `--fusion concat`; it underperformed
because the one-hot tags dominated the normalization and drowned out FastText.

## Project structure

```
data/
  raw/                    untouched FiReCS dataset (review, label)
  processed/              cleaned train / val / test CSVs
                          + cached calamanCy tags per tagger (*_tagged__<tagger>.pkl)
models/
  fasttext/cc.tl.300.bin  pretrained FastText Tagalog vectors (~4GB, downloaded once)
  <tagger>/               vocab + embedding matrix + tagsets, rebuilt on every run
notebooks/
  explore_data.ipynb      loads FiReCS, cleans text, creates the train/val/test split
results/
  development/<tagger>/   experiments used to CHOOSE the design (validation set)
  final/                  final thesis runs on the test set (created in the final round)
src/
  preprocessing.py        text cleaning + stratified train/val split
  tokenization.py         calamanCy tagging (tokens, POS, dep, morph, lemma)
  data_prep.py            loads splits + hash-validated tag caches
  vocab.py                vocabulary + FastText embedding matrix
  features.py             one-hot POS / dependency features
  feature_extraction.py   loads the FastText model
  dataset.py              PyTorch Dataset + padding
  model.py                the Bi-LSTM (baseline and proposed, all fusion options)
  train.py                training loop with early stopping
  evaluate.py             accuracy / precision / recall / F1 + confusion matrix
  run_sweep.py            MAIN SCRIPT: runs several seeds and compares the models
  run_experiment.py       one training run (quick checks)
  compare_configs.py      Phase 3: compares training-setting configs, applies the selection rule
  inspect_tags.py         compares calamanCy taggers' tag quality (no training)
  quiet.py                hides library noise, keeps terminal output live
```

`data/` and `models/` are excluded from git (large and regenerable).

## Pipeline order

1. `notebooks/explore_data.ipynb` → creates `data/raw/` and `data/processed/*.csv`
2. `src/run_sweep.py` → tags the text (cached after the first run), builds the
   embeddings, trains both models over several seeds, prints the comparison

## Running experiments

Run from the repo root with the `firecs` environment active.

```
python src/run_sweep.py                          # baseline + proposed, 5 seeds, validation set
python src/run_sweep.py --models proposed        # proposed only, paired against stored baseline
python src/run_sweep.py --fusion concat          # the original manuscript fusion
python src/run_experiment.py --model proposed    # a single training run
```

Defaults: tagger `tl_calamancy_md-0.2.0`, fusion `projection`, **validation set**,
learning rate **0.0005** (chosen in Phase 3), dropout 0.3, no input dropout, no weight decay.

**Phase 3: overfitting.** Both models memorize the training set within 2–3 epochs.
Each config below trains BOTH models with the same settings (5 seeds each); then
`compare_configs.py` picks the winner by a rule fixed in advance (highest mean score
of both models together, and it must beat the current settings by more than 0.003):

```
python src/run_sweep.py --dropout 0.5                              # C1
python src/run_sweep.py --input-dropout 0.25                       # C2
python src/run_sweep.py --weight-decay 0.01                        # C3
python src/run_sweep.py --lr 0.0005                                # C4
python src/run_sweep.py --input-dropout 0.25 --weight-decay 0.01   # C5
python src/compare_configs.py                                      # compare + apply the rule
```

**Phase 3 is finished: C4 (learning rate 0.0005) won and is now the default.** C0 (the
manuscript's settings, lr 0.001) results are in `results/development/<tagger>/`; every
other config is in `results/development/<tagger>/regularization/<config>/`.
Add `--lr 0.001` to reproduce the manuscript settings.

**Final round (test set, run once):**
```
python src/run_sweep.py --split test                                     # baseline + proposed, 30 fresh seeds (1000-1029)
python src/run_sweep.py --split test --resume                            # continue if interrupted
python src/run_sweep.py --split test --models proposed --fusion concat   # optional: manuscript design
```
Results go to `results/final/<tagger>/`. The script refuses test runs with non-chosen
settings or development seeds.

**Validation vs test — important:**
- Use the **validation** set (the default) for every design decision.
- Use `--split test` **only for the final thesis runs**, after the design and
  hyperparameters are frozen. Picking a design based on test scores would leak
  the test set and invalidate the comparison against Cosme & De Leon (2024).

Each run writes `results/development/<tagger>/<model>/seed_<n>/metrics_val.json`
(or `metrics.json` for the test set). Model weights are only saved with
`--save-checkpoint` (~20MB each).

## Setup Instructions

### 1. Install Anaconda
Download from https://www.anaconda.com/download and install with default options.

### 2. Clone this repo
```
git clone https://github.com/dwknesho/calamancy-bilstm-sentiment.git
cd calamancy-bilstm-sentiment
```

### 3. Create the conda environment
```
conda create -n firecs python=3.10
conda activate firecs
```

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Open in VS Code
- Install the **Python** and **Jupyter** extensions
- `Ctrl+Shift+P` → `Python: Select Interpreter` → choose the `firecs` environment
- Every terminal you open in VS Code afterward should show `(firecs)` at the prompt

### 6. Windows only — allow conda activation in PowerShell
If you see an error like "running scripts is disabled on this system" when
opening a terminal, run this once:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then open a fresh terminal.

### 7. Models (downloaded automatically, not in the repo)
- **FastText** (Tagalog, ~4GB): downloads to `models/fasttext/` the first time the
  code runs. To trigger it manually:
  ```python
  from src.feature_extraction import load_fasttext_model
  load_fasttext_model("tl")
  ```
- **calamanCy** (`tl_calamancy_md-0.2.0`): downloads automatically the first time
  it is used.

### 8. Dataset
The FiReCS dataset downloads automatically via HuggingFace's `datasets` library
the first time the notebook runs.
```python
from datasets import load_dataset
dataset = load_dataset("ccosme/FiReCS")
```

## Notes for contributors

- Never commit files inside `data/` or `models/` — they're gitignored on purpose.
- `requirements.txt` lists only the packages the code imports directly. When you add one,
  add its line by hand (version from `pip show <package>`) and save the file as UTF-8.
  Don't paste in `pip freeze` output: it includes `@ file:///...` paths from your own
  computer, which break `pip install -r` on everyone else's.
- Keep the baseline and proposed models on the same Bi-LSTM class and the same
  hyperparameters, so the comparison stays fair.
