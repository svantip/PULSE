"""
Download Kaggle dataset and save as raw.csv
"""

import kagglehub
import pandas as pd
import os
import shutil

# Download latest version
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download(
    "tobiasbueck/multilingual-customer-support-tickets")

print(f"Dataset downloaded to: {path}")

# Find CSV files in the downloaded path
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"\nFound {len(csv_files)} CSV files")

if csv_files:
    # Find the largest CSV file by row count
    largest_file = None
    max_rows = 0

    print("\nAnalyzing CSV files:")
    for csv_file in csv_files:
        file_path = os.path.join(path, csv_file)
        try:
            df_temp = pd.read_csv(file_path)
            row_count = len(df_temp)
            print(f"  {csv_file}: {row_count:,} rows")
            if row_count > max_rows:
                max_rows = row_count
                largest_file = csv_file
        except Exception as e:
            print(f"  {csv_file}: Error reading - {e}")

    if largest_file:
        print(f"\n📊 Using largest file: {largest_file} ({max_rows:,} rows)")
        source_file = os.path.join(path, largest_file)
        destination = "data/raw.csv"

        # Copy to data folder as raw.csv (overwrite if exists)
        shutil.copy2(source_file, destination)

        # Verify
        df = pd.read_csv(destination)
        print(f"\n✅ Dataset saved to {destination}")
        print(f"Total records: {len(df):,}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
    else:
        print("❌ Could not read any CSV files")
else:
    print("❌ No CSV files found in downloaded dataset")
