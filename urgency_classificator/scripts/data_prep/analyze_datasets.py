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

    # Analyze text lengths
    if 'body' in df.columns:
        df['body_length'] = df['body'].astype(str).str.len()
        df['body_words'] = df['body'].astype(str).str.split().str.len()
        print(f"  Text length stats (characters):")
        print(f"    Min: {df['body_length'].min()}")
        print(f"    Max: {df['body_length'].max():,}")
        print(f"    Mean: {df['body_length'].mean():.0f}")
        print(f"    Median: {df['body_length'].median():.0f}")
        print(f"    95th percentile: {df['body_length'].quantile(0.95):.0f}")
        print(f"    99th percentile: {df['body_length'].quantile(0.99):.0f}")
        print(f"  Text length stats (words):")
        print(f"    Max: {df['body_words'].max():,}")
        print(f"    Mean: {df['body_words'].mean():.0f}")
        print(f"    95th percentile: {df['body_words'].quantile(0.95):.0f}")

        # Approximate token count (rough estimate: ~1.3 tokens per word)
        approx_tokens = df['body_words'] * 1.3
        print(f"  Approximate tokens (1.3x words):")
        print(f"    Max: {approx_tokens.max():.0f}")
        print(f"    Mean: {approx_tokens.mean():.0f}")
        print(f"    95th percentile: {approx_tokens.quantile(0.95):.0f}")
        print(f"    99th percentile: {approx_tokens.quantile(0.99):.0f}")

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

# Recommended MAX_LENGTH based on data
print("\n" + "="*70)
print("MAX_LENGTH RECOMMENDATION")
print("="*70)

# Analyze ALL datasets combined (since merge is recommended)
all_dfs_with_lengths = []
for filename, df in datasets.items():
    if 'body_words' in df.columns:
        all_dfs_with_lengths.append(df[['body_words']])

if all_dfs_with_lengths:
    combined_df = pd.concat(all_dfs_with_lengths, ignore_index=True)
    approx_tokens = combined_df['body_words'] * 1.3
    p95 = approx_tokens.quantile(0.95)
    p99 = approx_tokens.quantile(0.99)
    p50 = approx_tokens.quantile(0.50)

    print(f"\nCOMBINED DATASET (all files merged):")
    print(f"  50% of texts fit in: ~{p50:.0f} tokens (median)")
    print(f"  95% of texts fit in: ~{p95:.0f} tokens")
    print(f"  99% of texts fit in: ~{p99:.0f} tokens")

    if p95 <= 128:
        recommended = 128
    elif p95 <= 256:
        recommended = 256
    elif p95 <= 512:
        recommended = 512
    else:
        recommended = 1024

    print(f"\n✅ RECOMMENDED MAX_LENGTH: {recommended}")
    print(f"   (covers 95% of merged data, faster training)")

    if p99 <= recommended:
        print(f"   ✓ Also covers 99% of data")
    else:
        coverage_at_recommended = (approx_tokens <= recommended).mean() * 100
        print(
            f"   ✓ Actual coverage: {coverage_at_recommended:.1f}% of all texts")
        print(
            f"   ⚠ To cover 99%, use MAX_LENGTH: {512 if p99 <= 512 else 1024}")

    # Show alternative options
    print(f"\n📊 ALTERNATIVE OPTIONS:")
    for max_len in [128, 256, 512]:
        coverage = (approx_tokens <= max_len).mean() * 100
        truncated = 100 - coverage
        print(
            f"   MAX_LENGTH={max_len}: {coverage:.1f}% coverage ({truncated:.1f}% truncated)")
