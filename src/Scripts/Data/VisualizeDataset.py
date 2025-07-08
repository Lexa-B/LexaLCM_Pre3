import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time
import argparse
import numpy as np
import gc

class VisualizeWikipedia:
    def __init__(self, parquet_dir: str | Path, sample: bool = False, batch_size: int = 10):
        self.parquet_dir = Path(parquet_dir)
        self.sample = sample
        self.batch_size = batch_size
        self.parquet_files = glob.glob(str(self.parquet_dir / "*.parquet"))
        print(f"Found {len(self.parquet_files)} parquet files")

    @staticmethod
    def read_parquet_file(file):
        return pd.read_parquet(file)

    def process_batch(self, files):
        with ProcessPoolExecutor() as executor:
            dfs = list(executor.map(self.read_parquet_file, files))
        return pd.concat(dfs, ignore_index=True)

    def visualize(self):
        if self.sample:
            sample_files = self.parquet_files[::10]
        else:
            sample_files = self.parquet_files
        print(f"Using {len(sample_files)} files for visualization")

        # Initialize statistics
        total_rows = 0
        total_sentences = 0
        text_lengths = []
        sentence_counts = []
        date_counts = {}
        sentence_lengths = []

        # Process files in batches
        start_time = time.time()
        for i in tqdm(range(0, len(sample_files), self.batch_size), desc="Processing batches"):
            try:
                batch_files = sample_files[i:i + self.batch_size]
                batch_df = self.process_batch(batch_files)
                
                # Update statistics
                total_rows += len(batch_df)
                
                if 'text_sentences' in batch_df.columns:
                    batch_sentence_lists = batch_df['text_sentences'].dropna()
                    batch_sentence_counts = batch_sentence_lists.apply(len)
                    total_sentences += batch_sentence_counts.sum()
                    sentence_counts.extend(batch_sentence_counts.tolist())

                    # Flatten and get sentence lengths
                    for sentence_list in batch_sentence_lists:
                        sentence_lengths.extend([len(s) for s in sentence_list if isinstance(s, str)])

                
                if 'text' in batch_df.columns:
                    text_lengths.extend(batch_df['text'].str.len().tolist())
                
                if 'timestamp' in batch_df.columns:
                    batch_df['timestamp'] = pd.to_datetime(batch_df['timestamp'])
                    batch_df['date'] = batch_df['timestamp'].dt.date
                    for date, count in batch_df['date'].value_counts().items():
                        date_counts[date] = date_counts.get(date, 0) + count

                # Clear memory after each batch
                del batch_df
                gc.collect()

            except Exception as e:
                print(f"\nError processing batch {i//self.batch_size + 1}: {str(e)}")
                continue

        print(f"\nProcessing took {time.time() - start_time:.2f} seconds")

        # Display basic information
        print("\nDataset Information:")
        print("=" * 50)
        print(f"Total number of rows: {total_rows:,}")
        print(f"Total number of sentences: {total_sentences:,}")

        # Get a sample of the data for detailed statistics
        sample_df = pd.read_parquet(sample_files[0])
        print("\nColumns in the dataset:")
        print(sample_df.columns.tolist())
        print("\nData types:")
        print(sample_df.dtypes)
        print("\nSample rows:")
        print("=" * 50)
        print(sample_df.head())
        del sample_df
        gc.collect()

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(24, 12))  # 2 rows, 2 columns
        axes = axes.flatten()  # Flatten the 2x2 array for easier indexing

        # 1. Text length distribution
        if text_lengths:
            sns.histplot(data=text_lengths, bins=range(0, 30001, 250), ax=axes[0])
            axes[0].set_title('Text Lengths')
            axes[0].set_xlabel('Characters')
            axes[0].set_ylabel('Count')
            axes[0].set_xlim(0, 30000)

        # 2. Article date distribution
        if date_counts:
            dates = sorted(date_counts.keys())
            counts = [date_counts[date] for date in dates]
            axes[1].plot(dates, counts)
            axes[1].set_title('Articles Over Time')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)

        # 3. Sentence count per article
        if sentence_counts:
            sns.histplot(data=sentence_counts, bins=range(0, 1001, 10), ax=axes[2])
            axes[2].set_title('Sentence Counts per Article')
            axes[2].set_xlabel('Sentence Count')
            axes[2].set_ylabel('Count')
            axes[2].set_xlim(0, 1000)

        # 4. Sentence length
        if sentence_lengths:
            sns.histplot(data=sentence_lengths, bins=range(0, 301, 2), ax=axes[3])
            axes[3].set_title('Sentence Lengths')
            axes[3].set_xlabel('Characters')
            axes[3].set_ylabel('Count')
            axes[3].set_xlim(0, 300)

        # Save the plot
        plt.tight_layout()
        plt.savefig('src/_TEMP/wikipedia_dataset_visualization.png')

        print("\nVisualization saved as 'wikipedia_dataset_visualization.png'")

        # Display basic statistics
        print("\nBasic Statistics:")
        print("=" * 50)
        if text_lengths:
            print(f"Text length - Mean: {np.mean(text_lengths):.2f}, Median: {np.median(text_lengths):.2f}")
            print(f"Text length - Min: {min(text_lengths):,}, Max: {max(text_lengths):,}")
            print(f"Text length - 25th percentile: {np.percentile(text_lengths, 25):.2f}")
            print(f"Text length - 75th percentile: {np.percentile(text_lengths, 75):.2f}")
        if sentence_counts:
            print(f"\nSentence count - Mean: {np.mean(sentence_counts):.2f}, Median: {np.median(sentence_counts):.2f}")
            print(f"Sentence count - Min: {min(sentence_counts):,}, Max: {max(sentence_counts):,}")
            print(f"Sentence count - 25th percentile: {np.percentile(sentence_counts, 25):.2f}")
            print(f"Sentence count - 75th percentile: {np.percentile(sentence_counts, 75):.2f}")
        if sentence_lengths:
            print(f"\nSentence length - Mean: {np.mean(sentence_lengths):.2f}, Median: {np.median(sentence_lengths):.2f}")
            print(f"Sentence length - Min: {min(sentence_lengths):,}, Max: {max(sentence_lengths):,}")
            print(f"Sentence length - 25th percentile: {np.percentile(sentence_lengths, 25):.2f}")
            print(f"Sentence length - 75th percentile: {np.percentile(sentence_lengths, 75):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Wikipedia dataset")
    parser.add_argument("-d", "--parquet_dir", type=str, required=True, help="Directory containing the parquet files")
    parser.add_argument("-s", "--sample", action="store_true", help="Use a sample of files for faster processing")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="Number of files to process at once")
    args = parser.parse_args()
    visualize_wikipedia = VisualizeWikipedia(parquet_dir=args.parquet_dir, sample=args.sample, batch_size=args.batch_size)
    visualize_wikipedia.visualize()