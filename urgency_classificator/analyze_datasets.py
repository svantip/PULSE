"""
Analyze and potentially merge all CSV files from the dataset
"""

import pandas as pd
import os

# Path to downloaded dataset
path = "/Users/svantipuric/.cache/kagglehub/datasets/tobiasbueck/multilingual-customer-support-tickets/versions/14"

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

print("="*70)
print("ANALYZING ALL DATASET FILES")
print("="*70)

datasets = {}
for csv_file in csv_files:
    file_path = os.path.join(path, csv_file)
    df = pd.read_csv(file_path)
    datasets[csv_file] = df
    print(f"\n{csv_file}:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    if 'priority' in df.columns:
        print(f"  Priority distribution:")
        print(f"    {df['priority'].value_counts().to_dict()}")

# Check for duplicates between datasets
print("\n" + "="*70)
print("CHECKING FOR OVERLAPS")
print("="*70)

# Check if smaller datasets are subsets of larger ones
largest_file = 'aa_dataset-tickets-multi-lang-5-2-50-version.csv'
largest_df = datasets[largest_file]

for filename, df in datasets.items():
    if filename != largest_file:
        # Check if this dataset's bodies are in the largest dataset
        overlap = df[df['body'].isin(largest_df['body'])]
        overlap_pct = (len(overlap) / len(df)) * 100
        print(f"\n{filename}:")
        print(
            f"  {len(overlap):,} / {len(df):,} rows ({overlap_pct:.1f}%) found in largest dataset")

# Decision
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

# Count unique bodies across all datasets
all_bodies = pd.concat([df['body'] for df in datasets.values()])
unique_bodies = all_bodies.drop_duplicates()
print(f"\nTotal rows across all files: {len(all_bodies):,}")
print(f"Unique records (by body): {len(unique_bodies):,}")
print(f"Largest file has: {len(largest_df):,}")

if len(unique_bodies) > len(largest_df) * 1.1:  # 10% more
    print("\n✅ MERGE RECOMMENDED: Files contain different data")
    print("   Action: Merge all files and remove duplicates")
else:
    print("\n✅ NO MERGE NEEDED: Largest file contains most/all data")
    print("   Action: Use largest file only")
