"""
Clean all_tickets.csv dataset for urgency classification.
Maps 4-level urgency (0,1,2,3) to 3-level (low, medium, high).
"""

import pandas as pd
import re
import os


def map_urgency_to_priority(urgency):
    """
    Map 4-level urgency (0,1,2,3) to 3-level priority (low, medium, high).

    Urgency mapping:
    - 0 (critical/emergency) → high
    - 1 (high) → high  
    - 2 (medium) → medium
    - 3 (low) → low
    """
    if urgency in [0, 1]:
        return 'high'
    elif urgency == 2:
        return 'medium'
    else:  # urgency == 3
        return 'low'


def clean_text(text):
    """Clean text while preserving important content."""
    if not isinstance(text, str):
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)

    # Strip
    text = text.strip()

    return text


def main():
    """Main cleaning function."""
    # Paths
    input_file = 'data/all_tickets.csv'
    output_file = 'data/clean.csv'

    print("="*70)
    print("CLEANING ALL_TICKETS.CSV DATASET")
    print("="*70)

    # Load data
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} tickets")

    # Select body and urgency columns
    print("\nExtracting body and urgency columns...")
    df_clean = df[['body', 'urgency']].copy()

    # Remove nulls
    initial_count = len(df_clean)
    df_clean = df_clean.dropna()
    df_clean = df_clean[df_clean['body'].str.strip() != '']
    print(f"Removed {initial_count - len(df_clean)} null/empty rows")

    # Map urgency to priority
    print("\nMapping urgency levels...")
    print("  0 (critical) → high")
    print("  1 (high) → high")
    print("  2 (medium) → medium")
    print("  3 (low) → low")
    df_clean['priority'] = df_clean['urgency'].apply(map_urgency_to_priority)

    # Clean text
    print("\nCleaning text...")
    df_clean['body'] = df_clean['body'].apply(clean_text)

    # Remove empty after cleaning
    df_clean = df_clean[df_clean['body'].str.len() > 10]  # At least 10 chars

    # Drop urgency column, keep only body and priority
    df_clean = df_clean[['body', 'priority']]

    # Remove duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['body'], keep='first')
    print(f"Removed {initial_count - len(df_clean)} duplicates")

    # Print statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total samples: {len(df_clean)}")

    print("\nPriority distribution:")
    priority_counts = df_clean['priority'].value_counts().sort_index()
    for priority, count in priority_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"  {priority}: {count} ({percentage:.2f}%)")

    print(f"\nText length statistics (characters):")
    text_lengths = df_clean['body'].str.len()
    print(f"  Mean: {text_lengths.mean():.2f}")
    print(f"  Median: {text_lengths.median():.2f}")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")

    # Save
    print(f"\nSaving to {output_file}...")
    df_clean.to_csv(output_file, index=False)
    print(f"Saved {len(df_clean)} tickets")

    print("\n" + "="*70)
    print("CLEANING COMPLETE!")
    print("="*70)

    return df_clean


if __name__ == "__main__":
    main()
