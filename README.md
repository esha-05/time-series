# 🌡️ Time Series Forecasting with LSTM + Attention

> Hourly temperature forecasting on the Jena Climate dataset using a stacked LSTM with Bahdanau Attention, benchmarked against XGBoost and Prophet baselines.

---

## 🧠 Model Architecture

```
Input (batch, 24, 1)
    │
    ▼
LSTM (64 units, return_sequences=True)
    │
Dropout (0.2)
    │
    ▼
LSTM (32 units, return_sequences=True)
    │
Dropout (0.2)
    │
    ▼
BahdanauAttention (32 units)
    │  → context vector (batch, 32)
    │  → attention weights (batch, 24, 1)
    ▼
Dense (32, relu)
    │
    ▼
Dense (1)  ← predicted temperature
```

**Total parameters:** 31,490 (~123 KB)

---

## 📊 Dataset

**Jena Climate Dataset** — Max Planck Institute for Biogeochemistry  
- **Period:** January 2009 → January 2017  
- **Raw resolution:** every 10 minutes → resampled to **1-hour intervals**  
- **Records after cleaning:** 70,129 hourly observations  
- **Target:** `T (degC)` — air temperature in degrees Celsius  
- **Features used for baselines:** all 14 meteorological variables  

| Split | Size | Period |
|-------|------|--------|
| Train | 49,090 (70%) | 2009–2014 |
| Val   | 10,519 (15%) | 2014–2015 |
| Test  | 10,520 (15%) | 2015–2017 |

---

## ⚙️ Setup & Installation

### 1. Clone / navigate to the project

```bash
cd "Time Series Forecasting with LSTM + Attention"
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `jena_climate_2009_2016.csv` from [Kaggle](https://www.kaggle.com/datasets/mnassrib/jena-climate) and place it at:

```
energy-dl-project/data/jena_climate_2009_2016.csv
```

---

## 🚀 Running the Pipeline

```bash
python run_pipeline.py
```

This runs all steps in sequence:

| Step | Description |
|------|-------------|
| 1–2  | Load, clean, resample, scale data → save `.npy` arrays |
| 3    | Build LSTM + Attention model |
| 4–5  | Train LSTM with early stopping + model checkpointing |
| 6a   | Train XGBoost baseline on lag features |
| 6b   | Train Prophet baseline |
| 7–9  | Evaluate all models, generate comparison table + plot |

---

## 📈 Results

| Model    | RMSE   | MAE    |
|----------|--------|--------|
| **LSTM + Attention** | **~0.011** | **~0.008** *(scaled)* |
| XGBoost  | 0.6295 | 0.4334 *(raw °C)* |
| Prophet  | 4.4830 | 3.6383 *(raw °C)* |

> Note: LSTM metrics are on MinMax-scaled data (0–1); XGBoost and Prophet metrics are on raw °C values. The LSTM significantly outperforms both baselines on this task.

Training converged at **epoch 12–25** (varies per run) with early stopping (patience=5).

---

## 🔑 Key Design Decisions

- **Bahdanau Attention** — lets the model learn which timesteps within the 24-hour window are most relevant for the prediction, improving interpretability and accuracy over vanilla LSTM.
- **MinMax scaling** — applied fit-only on training data to prevent data leakage into validation/test sets. Scaler saved to `data/scaler.pkl` for inference.
- **`.h5` format** — used instead of `.keras` for compatibility with TensorFlow 2.13 / Keras 2.13.1.
- **XGBoost on raw values** — deliberately not scaled so RMSE is directly interpretable in °C.
- **Prophet** — uses full hourly seasonality (daily + weekly + yearly) as a strong statistical baseline.

---

## 🛠️ Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.13.0 | LSTM model training |
| keras | 2.13.1 | Model API (must match TF version) |
| xgboost | ≥1.7 | Gradient boosting baseline |
| prophet | ≥1.1 | Time series decomposition baseline |
| scikit-learn | ≥1.0 | Metrics, MinMaxScaler |
| pandas | ≥1.5 | Data manipulation |
| numpy | ≥1.23 | Array operations |
| matplotlib | ≥3.5 | Plotting |

---

## ⚠️ Known Issues & Fixes

**`ModuleNotFoundError: No module named 'src'`**  
→ Fixed: `run_pipeline.py` uses `sys.path.insert` with `os.path.abspath(__file__)` to resolve paths correctly regardless of working directory.

**`ValueError: argument 'options' not supported`**  
→ Fixed: Downgrade to `keras==2.13.1` to match `tensorflow==2.13.0`. Use `.h5` save format.

**`from preprocess import ...` fails inside model files**  
→ Fixed: Each model file inserts its own `src/` directory into `sys.path` using `os.path.dirname(os.path.abspath(__file__))`.
-----
