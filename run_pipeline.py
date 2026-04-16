import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy-dl-project"))

import numpy as np
from src.preprocess import load_and_clean, split_and_scale, create_sequences
from src.model_lstm import train as train_lstm
from src.model_xgboost import train as train_xgb
from src.model_prophet import train as train_prophet
from src.evaluate import run_comparison

print("=" * 50)
print("STEP 1-2: Preprocessing")
print("=" * 50)

df = load_and_clean()
train_s, val_s, test_s, scaler = split_and_scale(df)
X_train, y_train = create_sequences(train_s)
X_val,   y_val   = create_sequences(val_s)
X_test,  y_test  = create_sequences(test_s)

os.makedirs("data", exist_ok=True)
np.save("data/X_train.npy", X_train); np.save("data/y_train.npy", y_train)
np.save("data/X_val.npy",   X_val);   np.save("data/y_val.npy",   y_val)
np.save("data/X_test.npy",  X_test);  np.save("data/y_test.npy",  y_test)

print("\n" + "=" * 50)
print("STEP 4-5: Train LSTM + Attention")
print("=" * 50)
model_lstm, history = train_lstm(epochs=30)

print("\n" + "=" * 50)
print("STEP 6a: Train XGBoost baseline")
print("=" * 50)
_, y_te_xgb, preds_xgb, _, _ = train_xgb()

print("\n" + "=" * 50)
print("STEP 6b: Train Prophet baseline")
print("=" * 50)
_, y_te_prophet, preds_prophet, _, _ = train_prophet()

print("\n" + "=" * 50)
print("STEP 7-9: Evaluate + Multi-step forecast")
print("=" * 50)

# Get LSTM predictions on test set
X_test  = np.load("data/X_test.npy")
y_test  = np.load("data/y_test.npy")
preds_lstm = model_lstm.predict(X_test).flatten()

# Align lengths (Prophet/XGBoost may have slightly different test sizes)
min_len = min(len(y_test), len(preds_lstm))
predictions_dict = {
    "LSTM":     (y_test[:min_len],       preds_lstm[:min_len]),
    "XGBoost":  (y_te_xgb,              preds_xgb),
    "Prophet":  (y_te_prophet,           preds_prophet),
}

results = run_comparison(predictions_dict)

print("\n" + "=" * 50)
print("ALL DONE. Check the models/ folder for:")
print("  lstm_best.h5             ← trained model")
print("  xgboost_model.pkl")
print("  prophet_forecast.csv")
print("  comparison_results.csv   ← RMSE / MAE / MAPE table")
print("  comparison_plot.png      ← actual vs predicted")
print("=" * 50)