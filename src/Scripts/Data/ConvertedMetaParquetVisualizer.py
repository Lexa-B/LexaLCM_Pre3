# ConvertedMetaParquetVisualizer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--before-path", type=str, required=True, help="Path to before parquet file")
parser.add_argument("-a", "--after-path", type=str, required=True, help="Path to after parquet file")
args = parser.parse_args()

before_path = args.before_path
after_path = args.after_path

before = pd.read_parquet(before_path)
after = pd.read_parquet(after_path)

print("=== BEFORE ===")
print(before.dtypes)
print(before.head())
emb_before = before["text_sentences_sonar_emb"].iloc[0]
print("First 2 embeddings:", emb_before[:2])
print("Type check:", type(emb_before), type(emb_before[0]) if isinstance(emb_before, list) else "N/A")
try:
    arr_before = np.stack([np.array(vec, dtype=np.float32) for vec in emb_before])
    print("Dtype check:", arr_before.dtype, "Shape:", arr_before.shape)
except Exception as e:
    print("Failed to process BEFORE:", str(e))

print("\n=== AFTER ===")
print(after.dtypes)
print(after.head())
emb_after = after["text_sentences_sonar_emb"].iloc[0]
print("First 2 embeddings:", emb_after[:2])
print("Type check:", type(emb_after), type(emb_after[0]) if isinstance(emb_after, list) else "N/A")
try:
    arr_after = np.stack([np.array(vec, dtype=np.float32) for vec in emb_after])
    print("Dtype check:", arr_after.dtype, "Shape:", arr_after.shape)
except Exception as e:
    print("Failed to process AFTER:", str(e))

try:
    # Determine shared vmin/vmax for consistent color scale
    combined = np.concatenate([arr_before[:32], arr_after[:32]], axis=0)
    vmin = np.min(combined)
    vmax = np.max(combined)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axes[0].imshow(arr_before[:32], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("Before Conversion")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(arr_after[:32], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("After Conversion")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("src/_TEMP/embedding_side_by_side.png")
    plt.close()
    print("✅ Saved side-by-side visualization to 'src/_TEMP/embedding_side_by_side.png'")
except Exception as e:
    print("❌ Visualization failed:", str(e))
