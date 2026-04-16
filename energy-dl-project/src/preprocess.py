import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle, os

RAW_PATH  = "energy-dl-project\data\jena_climate_2009_2016.csv"
OUT_DIR   = "data"
TARGET    = "T (degC)"          
WINDOW    = 24                 
HORIZON   = 1                   
TRAIN_PCT = 0.7
VAL_PCT   = 0.15


def load_and_clean():
    df = pd.read_csv(RAW_PATH)

    # Fix column names
    df.columns = [c.strip() for c in df.columns]

    # Parse datetime
    df["Date Time"] = pd.to_datetime(df["Date Time"], dayfirst=True)
    df = df.sort_values("Date Time").set_index("Date Time")

    # Replace bad sensor values
    df = df.replace(-9999.0, np.nan)

    # Resample: average every 6 rows = 1-hour intervals
    df = df.resample("1h").mean()

    # Forward-fill the small gaps left after resampling
    df = df.ffill().dropna()

    print(f"Shape after cleaning: {df.shape}")
    print(f"Date range: {df.index[0]}  →  {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    return df


def split_and_scale(df):
    series = df[TARGET].values.reshape(-1, 1)

    n = len(series)
    train_end = int(n * TRAIN_PCT)
    val_end   = int(n * (TRAIN_PCT + VAL_PCT))

    train = series[:train_end]
    val   = series[train_end:val_end]
    test  = series[val_end:]

    # fit scaler ONLY on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    val_scaled   = scaler.transform(val)
    test_scaled  = scaler.transform(test)

    print(f"\nSplit sizes → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Save scaler so we can inverse-transform predictions later
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return train_scaled, val_scaled, test_scaled, scaler


def create_sequences(data, window=WINDOW, horizon=HORIZON):
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i : i + window])
        y.append(data[i + window + horizon - 1, 0])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    df = load_and_clean()
    train_s, val_s, test_s, scaler = split_and_scale(df)

    X_train, y_train = create_sequences(train_s)
    X_val,   y_val   = create_sequences(val_s)
    X_test,  y_test  = create_sequences(test_s)

    np.save(f"{OUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/X_val.npy",   X_val)
    np.save(f"{OUT_DIR}/y_val.npy",   y_val)
    np.save(f"{OUT_DIR}/X_test.npy",  X_test)
    np.save(f"{OUT_DIR}/y_test.npy",  y_test)

    print("\nSaved arrays to data/")
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}    y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")