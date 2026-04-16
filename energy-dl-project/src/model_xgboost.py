import numpy as np
import pandas as pd
import pickle, os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def make_lag_features(series: np.ndarray, lags=(1, 2, 3, 6, 12, 24, 48)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.flatten()})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_mean_24"] = df["y"].shift(1).rolling(24).mean()
    df["rolling_std_24"]  = df["y"].shift(1).rolling(24).std()
    df = df.dropna()
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y


def train():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_and_clean, split_and_scale, TARGET

    df = load_and_clean()
    raw = df[TARGET].values

    n = len(raw)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_tr, y_tr = make_lag_features(raw[:train_end])
    X_te, y_te = make_lag_features(raw[train_end:val_end + (n - val_end)])

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        eval_metric="rmse",
        random_state=42,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=50,
    )

    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae  = mean_absolute_error(y_te, preds)
    print(f"\nXGBoost → RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    os.makedirs("models", exist_ok=True)
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, y_te, preds, rmse, mae


if __name__ == "__main__":
    train()