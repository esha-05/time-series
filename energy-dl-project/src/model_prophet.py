import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train():
    from preprocess import load_and_clean, TARGET

    df = load_and_clean()

    prophet_df = df[[TARGET]].reset_index()
    prophet_df.columns = ["ds", "y"]

    n = len(prophet_df)
    train_end = int(n * 0.70)

    train_df = prophet_df.iloc[:train_end]
    test_df  = prophet_df.iloc[train_end:]

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95,
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq="H")
    forecast = model.predict(future)

    test_forecast = forecast.iloc[train_end:].reset_index(drop=True)
    y_true = test_df["y"].values[:len(test_forecast)]
    y_pred = test_forecast["yhat"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"\nProphet → RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    os.makedirs("models", exist_ok=True)
    test_forecast.to_csv("models/prophet_forecast.csv", index=False)
    pd.DataFrame({"y_true": y_true}).to_csv("models/prophet_true.csv", index=False)

    return model, y_true, y_pred, rmse, mae


if __name__ == "__main__":
    train()