import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def load_duolingo_csv(path):
    df = pd.read_csv(path)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Binary correctness
    df["correct"] = (df["session_correct"] > 0).astype(int)

    # Rename item column
    df.rename(columns={"lexeme_id": "item_id"}, inplace=True)

    # Sort by user and time
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return df


def make_item_features(df):
    g = df.groupby("item_id")

    item_features = pd.DataFrame({
        "item_id": g.size().index,
        "seen_count": g["history_seen"].max().values,
        "correctness_rate": (g["history_correct"].max() / g["history_seen"].max()).fillna(0).values,
        "avg_delta": g["delta"].mean().fillna(0).values,
        "avg_p_recall": g["p_recall"].mean().values
    })

    item_features["difficulty"] = 1 - item_features["avg_p_recall"]

    item_features = item_features[
        ["item_id", "difficulty", "seen_count", "correctness_rate", "avg_delta", "avg_p_recall"]
    ]

    return item_features


def make_user_traces(df):
    user_traces = {}

    for user, user_df in df.groupby("user_id"):
        seq = []

        for _, row in user_df.iterrows():
            seq.append({
                "item_id": row["item_id"],
                "timestamp": row["timestamp"],
                "correct": int(row["correct"]),
                "delta": float(row["delta"]),
                "history_seen": int(row["history_seen"]),
                "history_correct": int(row["history_correct"])
            })
        user_traces[user] = seq

    return user_traces


def preprocess_and_save(csv_path, outdir="processed"):
    Path(outdir).mkdir(exist_ok=True)

    df = load_duolingo_csv(csv_path)
    item_features = make_item_features(df)
    user_traces = make_user_traces(df)

    # Save both
    with open(f"{outdir}/item_features.pkl", "wb") as f:
        pickle.dump(item_features, f)

    with open(f"{outdir}/user_traces.pkl", "wb") as f:
        pickle.dump(user_traces, f)

    print("Saved:")
    print(" - processed/item_features.pkl")
    print(" - processed/user_traces.pkl")

    return item_features, user_traces
