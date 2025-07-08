# VerifyEmbeddings.py
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

EMBEDDING_KEY = "text_sentences_sonar_emb"
EXPECTED_DIM = 1024


def verify_embedding_array(arr, expected_dim=EXPECTED_DIM):
    try:
        arr = np.stack([np.array(vec, dtype=np.float32) for vec in arr])
        if arr.ndim != 2:
            return False, f"Wrong ndim: {arr.ndim}"
        if arr.shape[1] != expected_dim:
            return False, f"Wrong dim: {arr.shape[1]}"
        return True, None
    except Exception as e:
        return False, str(e)


def scan_parquet_dir(directory):
    all_files = sorted(glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True))
    total_checked = 0
    total_failed = 0
    file_failures = {}

    for file in tqdm(all_files, desc="Scanning Parquet Files"):
        try:
            df = pd.read_parquet(file, columns=[EMBEDDING_KEY])
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")
            continue

        for i, row in enumerate(df[EMBEDDING_KEY]):
            total_checked += 1
            ok, reason = verify_embedding_array(row)
            if not ok:
                total_failed += 1
                file_failures.setdefault(file, []).append((i, reason))

    print("\n=== Scan Report ===")
    print(f"Files scanned: {len(all_files)}")
    print(f"Rows checked: {total_checked}")
    print(f"Broken rows: {total_failed}")

    if total_failed:
        print("\n🚨 Failures by file:")
        for file, failures in file_failures.items():
            print(f"{file}: {len(failures)} failures")
            for idx, reason in failures[:5]:  # only print first 5 per file
                print(f"  - Row {idx}: {reason}")
    else:
        print("✅ All embeddings look valid!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, required=True, help="Path to directory with parquet files")
    args = parser.parse_args()

    scan_parquet_dir(args.data_dir)
