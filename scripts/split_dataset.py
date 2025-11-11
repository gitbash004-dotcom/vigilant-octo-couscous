"""Utility script for splitting the HR analytics dataset into multiple files.

Example:
    python scripts/split_dataset.py --dataset data/hr_data_sample.csv --output-dir raw-data --num-files 4
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd


def chunk_dataframe(df: pd.DataFrame, num_chunks: int) -> List[pd.DataFrame]:
    """Split dataframe into ``num_chunks`` nearly even parts."""
    if num_chunks <= 0:
        raise ValueError("num_files must be positive")
    chunk_size = math.ceil(len(df) / num_chunks)
    return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the HR dataset into smaller raw files")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the input dataset CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder to store the split files")
    parser.add_argument("--num-files", required=True, type=int, help="Number of files to generate")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the rows before splitting. This is recommended to simulate streaming data.",
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset {args.dataset} does not exist")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    if args.shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    chunks = chunk_dataframe(df, args.num_files)

    for idx, chunk in enumerate(chunks, start=1):
        file_path = args.output_dir / f"raw_batch_{idx:03d}.csv"
        chunk.to_csv(file_path, index=False)
        print(f"Wrote {len(chunk)} rows to {file_path}")

    print(f"Generated {len(chunks)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
