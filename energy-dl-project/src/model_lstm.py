import numpy as np
import os
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(units)
        self.V = Dense(1)

    def call(self, encoder_outputs):
        score = self.V(tf.nn.tanh(self.W(encoder_outputs)))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * encoder_outputs
        context = tf.reduce_sum(context, axis=1)
        return context, weights


def build_model(window=24, features=1, lstm_units=64, attention_units=32, dropout=0.2):
    inputs = Input(shape=(window, features))

    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = Dropout(dropout)(x)

    x = LSTM(lstm_units // 2, return_sequences=True)(x)
    x = Dropout(dropout)(x)

    context, attn_weights = BahdanauAttention(attention_units)(x)

    x = Dense(32, activation="relu")(context)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model


def train(epochs=50, batch_size=64):
    # Resolve paths relative to this file so it works from any working directory
    src_dir     = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)          # energy-dl-project/
    root_dir    = os.path.dirname(project_dir)       # repo root

    data_dir   = os.path.join(root_dir, "data")
    models_dir = os.path.join(root_dir, "models")

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(data_dir, "y_val.npy"))

    model = build_model(
        window=X_train.shape[1],
        features=X_train.shape[2]
    )
    model.summary()

    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, "lstm_best.h5")

    checkpoint = ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    return model, history


if __name__ == "__main__":
    model, history = train()
    print("Training complete. Model saved to models/lstm_best.h5")