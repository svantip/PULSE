"""
Merge all CSV files and standardize to low/medium/high priority
"""

import pandas as pd
import os

path = "/Users/svantipuric/.cache/kagglehub/datasets/tobiasbueck/multilingual-customer-support-tickets/versions/14"
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

print("="*70)
print("MERGING ALL DATASETS")
print("="*70)

all_dfs = []

for csv_file in csv_files:
    file_path = os.path.join(path, csv_file)
    df = pd.read_csv(file_path)

    # Keep valuable metadata columns
    essential_cols = ['subject', 'body',
                      'priority', 'language', 'queue', 'type']
    df = df[[col for col in essential_cols if col in df.columns]]

    # Standardize priority to low/medium/high
    priority_mapping = {
        'very_low': 'low',
        'low': 'low',
        'medium': 'medium',
        'high': 'high',
        'critical': 'high'
    }

    if 'priority' in df.columns:
        df['priority'] = df['priority'].map(priority_mapping)

    print(f"\n{csv_file}: {len(df):,} rows")
    all_dfs.append(df)

# Merge all datasets
print("\n" + "="*70)
print("MERGING...")
merged_df = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows after merge: {len(merged_df):,}")

# Remove duplicates based on body
print("\nRemoving duplicates...")
merged_df = merged_df.drop_duplicates(subset=['body'], keep='first')
print(f"Unique rows: {len(merged_df):,}")

# Remove rows with missing body or priority (keep rows with missing metadata)
merged_df = merged_df.dropna(subset=['body', 'priority'])
print(f"After removing rows with missing body/priority: {len(merged_df):,}")

# Show final distribution
print("\n" + "="*70)
print("FINAL DATASET")
print("="*70)
print(f"Total records: {len(merged_df):,}")
print(f"\nPriority distribution:")
print(merged_df['priority'].value_counts())
print(f"\nPercentages:")
print(merged_df['priority'].value_counts(normalize=True) * 100)

# Combine subject + body for richer context
merged_df['body'] = merged_df.apply(
    lambda row: f"Betreff: {row['subject']}\\n{row['body']}"
    if 'subject' in merged_df.columns and pd.notna(row.get('subject')) else row['body'],
    axis=1
)

# Reorder columns - body and priority first, then metadata
cols = ['body', 'priority']
metadata_cols = [col for col in merged_df.columns if col not in [
    'body', 'priority', 'subject']]
final_df = merged_df[cols + metadata_cols]

# Save to raw.csv (overwrite)
output_path = "data/raw.csv"
final_df.to_csv(output_path, index=False)

print(f"\n✅ Merged dataset saved to {output_path}")
print(f"\nColumns: {list(final_df.columns)}")
print(f"\nMetadata availability:")
for col in metadata_cols:
    non_null = final_df[col].notna().sum()
    pct = (non_null / len(final_df)) * 100
    print(f"  {col}: {non_null:,} ({pct:.1f}%)")
print(f"\nSample records:")
print(final_df.head(3))
