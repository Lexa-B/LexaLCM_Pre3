# ConvertMetaParquet.py

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, required=True, help="Name of the dataset")
parser.add_argument("-i", "--input-dir", type=str, required=True, help="Path to directory with the dataset")
parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to directory to save the converted dataset")
args = parser.parse_args()


INPUT_DIR = args.input_dir + "/" + args.name
OUTPUT_DIR = args.output_dir + "/" + args.name

EMB_COL = "text_sentences_sonar_emb"
TEXT_SENT_COL = "text_sentences"
EXPECTED_DIM = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_embedding(raw):
    try:
        arr = np.stack([np.array(vec, dtype=np.float32) for vec in raw])
        if arr.ndim == 2 and arr.shape[1] == EXPECTED_DIM:
            return arr.tolist()
    except Exception:
        pass
    return None

def convert_parquet(in_path, out_path):
    try:
        df = pd.read_parquet(in_path)
    except Exception as e:
        print(f"❌ Failed to read {in_path}: {e}")
        return False

    if EMB_COL not in df.columns:
        print(f"⚠️  Skipping {in_path}: no '{EMB_COL}' column")
        return False

    cleaned_rows = []
    kept, dropped = 0, 0
    for idx, row in df.iterrows():
        emb = parse_embedding(row.get(EMB_COL))
        if emb is None:
            dropped += 1
            continue

        new_row = row.to_dict()
        new_row[EMB_COL] = emb
        cleaned_rows.append(new_row)
        kept += 1

    print(f"📄 {os.path.basename(in_path)}: kept {kept}, dropped {dropped} rows")

    if not cleaned_rows:
        print(f"⚠️  No valid rows in {in_path}")
        return False

    out_df = pd.DataFrame(cleaned_rows)
    out_df[EMB_COL] = out_df[EMB_COL].apply(lambda row: np.array(row, dtype=np.float32).tolist())

    if 'split' in out_df.columns:
        out_df = out_df.drop(columns=['split'])

    out_df.to_parquet(out_path, engine="pyarrow")
    return True

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    success, failed = 0, 0

    for file in tqdm(files, desc="Converting Parquet Files"):
        base = os.path.basename(file)
        out_file = os.path.join(OUTPUT_DIR, base)
        if convert_parquet(file, out_file):
            success += 1
        else:
            failed += 1

    print("\n=== Conversion Summary ===")
    print(f"✅ Converted: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output dir: {OUTPUT_DIR}")
