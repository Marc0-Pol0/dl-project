# DL Project: Predicting Earning Announcements Day Direction

Predicting stock price reactions to earnings announcements by combining firm fundamentals, market data and FinBERT-based news sentiment using deep learning models.

## Overview
- Task: multi-class classification (Up / Neutral / Down stock price movement after an earnings announcement)
- Dataset: custom-built dataset of ~1,000 US equity earnings announcements (Oct 2024 – Aug 2025), combining firm fundamentals, market prices and FinBERT-based news sentiment over a 30-day pre-announcement window
- Model: MLP baseline, LSTM and Transformer (TODO: eventually add other models)
- Goal / metric: predict post-EA price movement using accuracy, precision, recall, F1-score

## Repository Structure
- dataset_generation/  
  - `DL_project_data.ipynb` – Jupyter notebook used to build the custom dataset by aggregating firm fundamentals, market prices and news sentiment  
  - `DL_project_news.py` – merges previously downloaded news files (segmented by time periods and company batches) into a consolidated news dataset  
  - `utils.py` – helper functions  
  - `DL_project_config.yml` – configuration file for dataset construction  
  - `DL_dataset.pkl` – dataset (~1,000 earnings announcements) saved as a Python pickle file
  > Note: dataset generation requires private API access and was executed on an external Azure VM; paths and credentials are not portable or included in this repository.

(TODO: complete and adjust once the final repository is structured)

- src/        core code (models, training, evaluation)
- configs/    configuration files (yaml/json)
- scripts/    helper scripts (data prep, train, eval)
- notebooks/  experiments and analysis
- data/       datasets (usually not tracked)
- results/    logs, checkpoints, metrics, figures
- requirements.txt  dependencies

## Quickstart
1. Create environment (optional)
   python -m venv .venv  
   source .venv/bin/activate  (macOS/Linux)  
   .venv\Scripts\activate     (Windows)

2. Install dependencies  
   pip install -r requirements.txt (TODO)

3. Prepare data 
   Dataset was generated offline using `dataset_generation/DL_project_data.ipynb`
   and saved as `DL_dataset.pkl` (generation requires private API access).

4. Train (TODO)
   python src/train.py --config configs/train.yaml

5. Evaluate (TODO)
   python src/eval.py --config configs/eval.yaml

## Setup
Requirements:
- Python ≥ 3.9
- PyTorch (for LSTM/Transformer training)
- Hugging Face Transformers (FinBERT sentiment features)
- scikit-learn + scipy (preprocessing and metrics)
- Optional: CUDA-enabled GPU for faster training

Install (if `requirements.txt` is provided):
- pip install -r requirements.txt

## Data
- **Source:** `data/raw/DL_dataset.pkl` (provided; dataset generation is not reproducible from this repository)
- **Pipeline:** starting from `DL_dataset.pkl`, the preprocessing code in `src/data/` produces reproducible intermediate and final datasets
- **Folder structure:**
  - `data/raw/` – raw dataset (`DL_dataset.pkl`)
  - `data/processed/` – intermediate processed data
  - `data/trainable/` – final data used for model training
- **Preprocessing code:** `src/data/merge.py`, `src/data/preprocess.py`, `src/data/sentiment.py`

### Dataset schema (summary)
`DL_dataset.pkl` is a Python dictionary indexed by firm identifier.  
Each entry contains ticker mapping, prices, fundamentals, ratios, news, and earnings data
stored as pandas DataFrames. See the last cell of `dataset_generation/DL_project_data.ipynb` for the full schema.

## Configuration
No centralized configuration system is used.  
Model and preprocessing parameters are defined directly in the code.

## Usage
Training:
- python src/models/train.py

Evaluation:
- python src/models/evaluate.py

Inference:
- No standalone inference script is provided; predictions are produced as part of training and evaluation.

## Reproducibility
- Random seed: (TODO: a fixed seed may be used in the code; needs verification)
- Library versions: see `requirements.txt`
- Hardware: runs on GPU if CUDA is available, otherwise on CPU 
- Data: results are reproducible starting from `data/raw/DL_dataset.pkl` (dataset generation itself is not reproducible)
- Steps:
  1. Preprocess data starting from `DL_dataset.pkl`
  2. Train model
  3. Run evaluation

## Results
- **Saved outputs:** best-performing model checkpoints are stored in `networks/` (TODO: not sure if only the best version or all the checkpoints, check it)
- **Metrics:** classification metrics (e.g. accuracy, precision, recall, F1) are computed during evaluation but not saved
- **Models evaluated:** Logistic Regression, XGBoost, MLP, LSTM and Transformer
- **Summary:** (TODO: add a brief comparison of model performance)

## Contributors
- Manuel Noseda
- Nathan Soldati
- Marco Paina