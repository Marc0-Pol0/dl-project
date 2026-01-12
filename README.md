# DL Project: Predicting Earnings Announcement Day Direction

Predicting stock price reactions to earnings announcements by combining firm fundamentals, market data and FinBERT-based news sentiment using deep learning models.

## Overview
- Task: multi-class classification (Up / Neutral / Down stock price movement after an earnings announcement)
- Dataset: custom-built dataset of ~1,000 US equity earnings announcements (Oct 2024 – Aug 2025, 500 companies), combining firm fundamentals, market prices and FinBERT-based news sentiment over a 30-day pre-announcement window
- Model: MLP baseline, LSTM and Transformer
- Goal / metric: predict post-EA price movement using accuracy, precision, recall, F1-score

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
      - `utils.py` - helper functions
- data/
   - raw/ - raw data from the pipeline inside dataset_generation/
   - processed/ - cleaned data
   - trainable/ - cleaned and date-aligned data for top20 and top500 companies from the US market. Ready to use for training.
- networks/ - weights of trained networks
   - `lstm.pth` - baseline LSTM model
   - `transformer.pth` - baseline transformer model
   - `attention_buffer_ea_date.pth` - transformer model with 2 days buffer on EA date 
   - `lstm_buffer_ea_date.pth` - LSTM model with 2 days buffer on EA date 
   - `lstm_best.pth` - LSTM model with weighted Cross Entropy Loss
   - `lstm_sgd.pth` - LSTM model with SGD optimizer
- tests/
   - `playground.py` - playground file for data inspection
- `requirements.txt` - dependencies
- `README.md` - this file
- `runner.sh` - used to train models on the ETH student cluster

## Quickstart
1. Create **environment** (optional)
   python -m venv .venv  
   source .venv/bin/activate  (macOS/Linux)  
   .venv\Scripts\activate     (Windows)

2. Install **dependencies**  
   pip install -r requirements.txt

3. Prepare **data**: 
   Dataset was generated offline using `dataset_generation/DL_project_data.ipynb`
   and saved as `DL_dataset.pkl` (generation requires private API access). All the data necessary data is saved under the data repository.

4. **Train**: Configure `MODEL_SAVE_PATH` and run the file `src/train.py`. Trained model is saved to the `networks/` directory. On the ETH student cluster, run _sbatch runner.sh_.


5. **Evaluate**: Comment `run_training()` in `src/train.py` and run the script. The models analyzed in the report are the following: `lstm_buffer_ea_date` and `attention_buffer_ea_date`. Select `MODEL_SAVE_PATH` in the configuration class accordingly. You can also evaluate different pre-trained models from the `networks/` directory. 

## Setup
Requirements:
- Python ≥ 3.9
- PyTorch (for LSTM/Transformer training)
- Hugging Face Transformers (FinBERT sentiment features)
- scikit-learn + scipy (preprocessing and metrics)
- Optional: CUDA-enabled GPU for faster training

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
**Training** and **Evaluation**:
- python src/models/train.py

**Inference**:
- No standalone inference script is provided; predictions are produced as part of training and evaluation.

## Reproducibility
- **Library** versions: see `requirements.txt`
- **Hardware**: runs on GPU if CUDA is available, otherwise on CPU 
- **Data**: results are reproducible starting from `data/raw/DL_dataset.pkl` (dataset generation itself is not reproducible)
- **Steps**:
  1. Preprocess data starting from `DL_dataset.pkl`
  2. Train model
  3. Run evaluation

## Results
- **Saved outputs:** best-performing model checkpoints are stored in `networks/`.
- **Metrics:** classification metrics (e.g. accuracy, precision, recall, F1) are computed during evaluation
- **Models evaluated:** Logistic Regression, XGBoost, MLP, LSTM and Transformer
- **Summary:** Our evaluation shows that the LSTM acts as a conservative, risk-averse predictor, prioritizing precision by defaulting to a neutral stance during periods of high uncertainty. In contrast, the Transformer is more sensitive to market shocks, successfully identifying a higher volume of significant price movements through its attention mechanism. Both models demonstrate that fusing FinBERT sentiment with firm fundamentals creates a robust signal for distinguishing between stationary and volatile market states. Ultimately, the choice between architectures represents a trade-off between the LSTM's reliability and the Transformer's ability to capture stock volatility.

  > Note: All the details are presented in the report.

## Contributors
- Manuel Noseda
- Nathan Soldati
- Marco Paina