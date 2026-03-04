import os
import glob
import pandas as pd
import numpy as np


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["Date"])
    df["v"] = df["Volume"] / df["Volume"].shift(1) - 1.0
    df["cc"] = df["Close"] / df["Close"].shift(1) - 1.0
    df["co"] = (df["Open"] / df["Close"].shift(1)) - 1.0
    df["oc"] = (df["Close"] / df["Open"]) - 1.0
    return df[["Date", "cc", "co", "oc", "v"]]


def process_all(source_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(source_dir, "*.csv")))
    for path in paths:
        try:
            df = pd.read_csv(path)
            df = df.dropna()
            out = compute_returns(df)
            fname = os.path.basename(path)
            out_path = os.path.join(output_dir, fname)
            out.to_csv(out_path, index=False)
            print(f"saved: {out_path}")
        except Exception as e:
            print(f"failed on {path}: {e}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    source = os.path.join(base, "data", "source")
    output = os.path.join(base, "data", "process")
    process_all(source, output)
