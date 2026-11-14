"""Training and batch scoring for labeled customer feedback."""
import numpy as np
import pandas as pd
import tensorflow as tf


def clean_feedback(frame: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if text_col not in frame:
        raise ValueError(f"Missing text column: {text_col}")
    result = frame.copy()
    result[text_col] = result[text_col].fillna("").astype(str).str.strip()
    return result.loc[result[text_col].ne("")].reset_index(drop=True)


def train_model(frame: pd.DataFrame, text_col: str, label_col: str, epochs: int = 8):
    data = clean_feedback(frame, text_col)
    if label_col not in data:
        raise ValueError(f"Missing label column: {label_col}")
    labels = data[label_col].astype(str).str.strip().str.lower()
    allowed = {"negative", "neutral", "positive"}
    if len(data) < 12 or not allowed.issubset(set(labels)):
        raise ValueError("Provide at least 12 nonempty rows covering negative, neutral and positive labels.")
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    x = data[text_col].to_numpy(dtype=str)
    y = labels.map(mapping).to_numpy(dtype=np.int32)
    tf.keras.utils.set_random_seed(42)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=80)
    vectorizer.adapt(x)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(), dtype=tf.string), vectorizer,
        tf.keras.layers.Embedding(10000, 32),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x, y, epochs=epochs, batch_size=16, verbose=0, validation_split=0.2)
    return model, history.history


def predict_feedback(model, frame: pd.DataFrame, text_col: str) -> pd.DataFrame:
    data = clean_feedback(frame, text_col)
    if data.empty:
        raise ValueError("No nonempty feedback rows were found.")
    probabilities = model.predict(data[text_col].to_numpy(dtype=str), verbose=0)
    labels = np.array(["negative", "neutral", "positive"])
    data["predicted_sentiment"] = labels[np.argmax(probabilities, axis=1)]
    data["confidence"] = np.max(probabilities, axis=1).round(3)
    return data

