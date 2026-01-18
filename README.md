# DL Project: Predicting Earnings Announcement Day Direction

This project was developed as part of a Deep Learning course project. It is about predicting stock price reactions to earnings announcements by combining firm fundamentals, market data and FinBERT-based news sentiment using deep learning models.

## Overview
- Task: multi-class classification (Up / Neutral / Down stock price movement after an earnings announcement)
- Dataset: custom-built dataset of ~1,000 US equity earnings announcements (Oct 2024 – Aug 2025, 500 companies), combining firm fundamentals, market prices and FinBERT-based news sentiment over a 30-day pre-announcement window
- Models: Logistic Regression baseline, LSTM and Transformer
- Goal: predict post-EA price movement
- Metrics: Accuracy, Precision, Macro-F1, and a custom cost metric that penalizes incorrect Up/Down predictions more heavily than Neutral errors

## Repository Structure
- dataset_generation/  
  - `DL_project_data.ipynb` – Jupyter notebook used to build the custom dataset by aggregating firm fundamentals, market prices and news sentiment  
  - `DL_project_news.py` – merges previously downloaded news files (segmented by time periods and company batches) into a consolidated news dataset  
  - `utils.py` – helper functions  
  - `DL_project_config.yml` – configuration file for dataset construction  
  > Note: dataset generation requires private API access and was executed on an external Azure VM; paths and credentials are not portable or included in this repository.
- src/
   - data/
      - `merge.py` - merge fundamentals, news sentiment and stock values
      - `preprocess.py` - cleaning and aligning of raw data
      - `sentiment.py` - generate FinBERT sentiment distribution
   - figures/ - confusion matrices for all the models
   - models/
      - `dataloaders.py` - dataloaders for training and testing
      - `model.py` - classes for all the implemented models
      - `train.py` - training Pipeline
   - training/ - legacy folder with different models tested during the course of the project
- data/
   - raw/ - raw data from the pipeline inside dataset_generation/, not available in this repo for size reasons
   - processed/ - cleaned data
   - trainable/ - cleaned and date-aligned data for top20 and top500 companies from the US market. Ready to use for training.
- networks/ - weights of trained networks
   - `lstm.pth` - trained LSTM model
   - `attention.pth` - trained transformer model
   - `logreg.joblib` - trained logistic regression model
   - `lstm_nosent.pth` - LSTM model trained without sentiment analysis data
   - `attention_nosent.pth` - transformer model trained without sentiment analysis data
   - `logreg_nosent.joblib` - logistic regression model trained without sentiment analysis data
- tests/
   - `playground.py` - playground file for data inspection
- `requirements.txt` - dependencies
- `README.md` - this file
- `runner.sh` - used to train models on the ETH student cluster

## Quickstart
1. Create virtual **environment** (optional)

2. Install **dependencies**  

3. Prepare **data**: 
   Dataset was generated offline using `dataset_generation/DL_project_data.ipynb`
   and saved as `DL_dataset.pkl` (generation requires private API access). All the data necessary data is saved under the data repository.

4. **Train**: Choose the model you want to train and whether to use the sentiment data in the Config and run the file `src/train.py`. Trained model is saved to the `networks/` directory. On the ETH student cluster, run _sbatch runner.sh_.

5. **Evaluate**: The trained model will automatically be evaluated. The models analyzed in the report are the following: `lstm.pth`, `lstm_nosent.pth`, `attention.pth`, `attention_nosent.pth`, `logreg.joblib`, and `logreg_nosent.joblib`. You can also evaluate different pre-trained models from the `networks/` directory, by having the variable `REDO_TRAINING_IF_EXISTS` in the Config set to false and the right parameters.

## Setup
Requirements:
- Python ≥ 3.9
- PyTorch (for LSTM/Transformer training)
- Hugging Face Transformers (FinBERT sentiment features)
- scikit-learn + scipy (preprocessing and metrics)
- Optional: CUDA-enabled GPU for faster training

## Data
- **Source:** `data/raw/DL_dataset.pkl` (not provided in this repo for size reasons)
- **Pipeline:** starting from `DL_dataset.pkl`, the preprocessing code in `src/data/` produces reproducible intermediate and final datasets
- **Folder structure:**
  - `data/raw/` – raw dataset (`DL_dataset.pkl`), not provided in this repo
  - `data/processed/` – intermediate processed data
  - `data/trainable/` – final data used for model training
- **Preprocessing code:** `src/data/merge.py`, `src/data/preprocess.py`, `src/data/sentiment.py`

### Dataset schema (summary)
`DL_dataset.pkl` is a Python dictionary indexed by firm identifier.  
Each entry contains ticker mapping, prices, fundamentals, ratios, news, and earnings data
stored as pandas DataFrames. See the last cell of `dataset_generation/DL_project_data.ipynb` for the full schema.

## Configuration
No centralized configuration system is used.  
Model and preprocessing parameters are defined directly in the code. Key training options (model type, sentiment usage, early stopping) are controlled via the `Config` class in `src/models/train.py`.

## Usage
**Training** and **Evaluation**:
- python src/models/train.py

**Inference**:
- No standalone inference script is provided; predictions are produced as part of training and evaluation.

## Reproducibility
- **Hardware**: runs on GPU if CUDA is available, on MPS on MacOS if available, and on CPU otherwise.
- **Data**: results are reproducible starting from `data/raw/DL_dataset.pkl` (dataset generation itself is not reproducible)
- **Steps**:
  1. Preprocess data
  2. Train model
  3. Run evaluation

## Results
- **Saved outputs:** best-performing model checkpoints are stored in `networks/`.
- **Metrics:** classification metrics (e.g. accuracy, precision, recall, F1) are computed during evaluation
- **Models evaluated:** Logistic Regression, LSTM and Transformer
- **Summary:** Our results show that the LSTM achieves lower financial risk through conservative predictions, while the Transformer captures more volatile movements and attains a higher Macro-F1 score. The baseline performs poorly in terms of risk control, and sentiment features consistently improve stability and performance across models. These findings underscore the inherent trade-offs between sensitivity and risk management when applying deep learning models to noisy, imbalanced financial prediction tasks.

  > Note: All the details are presented in the report.

## Contributors
- Manuel Noseda
- Nathan Soldati
- Marco Paina
