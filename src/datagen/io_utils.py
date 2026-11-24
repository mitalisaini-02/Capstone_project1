"""
Deterministic, overwrite-protected writers.

gzip output uses mtime=0 and an empty embedded filename so identical data
produce byte-identical files (required for the reproducibility hash check).
"""
import gzip
import io
import json
import os

import pandas as pd


def write_csv_gz(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with open(path, "xb") as raw:                       # 'x' = refuse to overwrite
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="", compresslevel=6) as gz:
            gz.write(payload)


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x", newline="") as f:
        f.write(df.to_csv(index=False, lineterminator="\n"))


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_default)
        f.write("\n")


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x") as f:
        f.write(text)


def read_csv_gz(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
