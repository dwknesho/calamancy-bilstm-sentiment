# Enhancing Taglish Sentiment Classification with calamanCy + Bi-LSTM

Thesis project comparing a FastText-only baseline Bi-LSTM against a proposed
calamanCy-enhanced Bi-LSTM for Taglish (Tagalog-English) code-switched sentiment
classification, benchmarked against Cosme & De Leon (2024) on the FiReCS dataset.

## Project Structure

```
data/
  raw/            # untouched FiReCS dataset, never edited
  processed/      # cleaned train/val/test CSVs after preprocessing
src/              # reusable code (preprocessing, feature extraction, models, training)
notebooks/        # exploratory / scratch Jupyter notebooks
models/           # saved trained model weights (.pt files)
results/          # evaluation metrics, plots, confusion matrices
```

`data/`, `models/`, and downloaded model files are excluded from git via
`.gitignore` — they're large and regenerable, not stored in this repo.

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

### 7. Download required models (not included in repo — too large for git)
These download automatically the first time the code runs, but you can also
trigger them manually:

FastText (Tagalog, ~7GB):
```python
from src.feature_extraction import load_fasttext_model
load_fasttext_model("tl")  # downloads cc.tl.300.bin
```

calamanCy (Tagalog NLP pipeline) — downloads automatically the first time
`calamancy.load("tl_calamancy_md-0.1.0")` is called in code.

### 8. Dataset
The FiReCS dataset downloads automatically via HuggingFace's `datasets` library
the first time the notebook/scripts run — no manual download needed.
```python
from datasets import load_dataset
dataset = load_dataset("ccosme/FiReCS")
```

## Notes for contributors

- Never commit files inside `data/` or `models/` — they're gitignored on purpose.
- Run `pip freeze > requirements.txt` and commit it whenever you add a new package,
  so everyone's environment stays in sync.
- Baseline model = FastText embeddings only. Proposed model = FastText + calamanCy
  POS/dependency features fused together. Keep both using the same Bi-LSTM class
  and same hyperparameters so the comparison stays fair.