import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def run_comparison(predictions_dict):
    """
    predictions_dict: {
        'LSTM':    (y_true, y_pred),
        'XGBoost': (y_true, y_pred),
        'Prophet': (y_true, y_pred),
    }
    """
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"{'Model':<12} {'RMSE':>10} {'MAE':>10} {'MAPE%':>10}")
    print("-" * 45)

    rows = []
    all_results = {}

    for model_name, (y_true, y_pred) in predictions_dict.items():
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        mape_val = mape(y_true, y_pred)

        print(f"{model_name:<12} {rmse:>10.4f} {mae:>10.4f} {mape_val:>10.2f}")
        rows.append({"Model": model_name, "RMSE": rmse, "MAE": mae, "MAPE%": mape_val})
        all_results[model_name] = {"RMSE": rmse, "MAE": mae, "MAPE%": mape_val,
                                   "y_true": y_true, "y_pred": y_pred}

    print("=" * 50)

    # Save CSV
    os.makedirs("models", exist_ok=True)
    pd.DataFrame(rows).to_csv("models/comparison_results.csv", index=False)
    print("Saved → models/comparison_results.csv")

    # Save plot
    fig, axes = plt.subplots(len(predictions_dict), 1,
                             figsize=(14, 4 * len(predictions_dict)), sharex=False)
    if len(predictions_dict) == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, all_results.items()):
        plot_len = min(500, len(res["y_true"]))
        ax.plot(res["y_true"][:plot_len], label="Actual",    linewidth=1.5)
        ax.plot(res["y_pred"][:plot_len], label="Predicted", linewidth=1.5, alpha=0.8)
        ax.set_title(f"{model_name}  —  RMSE: {res['RMSE']:.4f}  MAE: {res['MAE']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("models/comparison_plot.png", dpi=150)
    plt.close()
    print("Saved → models/comparison_plot.png")

    return all_results