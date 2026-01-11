"""
Data cleaning script for urgency classification dataset.
Prepares data for BERT fine-tuning by keeping only body and priority columns,
removing null values, and performing standard text cleaning.
"""

import pandas as pd
import re
import os


def load_data(file_path):
    """
    Load the raw CSV dataset.

    Args:
        file_path (str): Path to the raw CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def select_relevant_columns(df):
    """
    Select only the body and priority columns needed for training.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with only body and priority columns
    """
    print("Selecting body and priority columns...")
    if 'body' not in df.columns or 'priority' not in df.columns:
        raise ValueError(
            "Required columns 'body' and 'priority' not found in dataset")

    df_clean = df[['body', 'priority']].copy()
    print(f"Selected columns: {list(df_clean.columns)}")
    return df_clean


def remove_null_values(df):
    """
    Remove rows with null or empty values in body or priority columns.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe without null values
    """
    print("Removing null values...")
    initial_count = len(df)

    # Remove rows where body or priority is null
    df_clean = df.dropna(subset=['body', 'priority'])

    # Remove rows where body or priority is empty string or whitespace only
    df_clean = df_clean[df_clean['body'].str.strip() != '']
    df_clean = df_clean[df_clean['priority'].str.strip() != '']

    removed_count = initial_count - len(df_clean)
    print(f"Removed {removed_count} rows with null or empty values")
    print(f"Remaining rows: {len(df_clean)}")

    return df_clean


def clean_text_for_bert(text):
    """
    Clean text data for BERT fine-tuning.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove excessive whitespace (tabs, newlines, multiple spaces)
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    # Remove any null byte characters
    text = text.replace('\x00', '')

    # BERT can handle most special characters, punctuation, and unicode
    # so we keep them for better semantic understanding

    return text


def normalize_priority_labels(df):
    """
    Normalize priority labels to ensure consistency.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with normalized priority labels
    """
    print("Normalizing priority labels...")

    # Convert to lowercase and strip whitespace
    df['priority'] = df['priority'].str.lower().str.strip()

    # Check unique priority values
    unique_priorities = df['priority'].unique()
    print(f"Unique priority values: {sorted(unique_priorities)}")

    # Validate that we only have expected priority values
    valid_priorities = {'low', 'medium', 'high'}
    invalid_priorities = set(unique_priorities) - valid_priorities

    if invalid_priorities:
        print(
            f"Warning: Found unexpected priority values: {invalid_priorities}")
        # Remove rows with invalid priorities
        df = df[df['priority'].isin(valid_priorities)]
        print(
            f"Removed rows with invalid priorities. Remaining rows: {len(df)}")

    return df


def apply_text_cleaning(df):
    """
    Apply text cleaning to the body column.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with cleaned text
    """
    print("Cleaning text in body column...")
    df['body'] = df['body'].apply(clean_text_for_bert)

    # Remove any rows where cleaning resulted in empty strings
    initial_count = len(df)
    df = df[df['body'].str.len() > 0]
    removed_count = initial_count - len(df)

    if removed_count > 0:
        print(f"Removed {removed_count} rows with empty body after cleaning")

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows based on body text.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe without duplicates
    """
    print("Removing duplicates...")
    initial_count = len(df)

    df_clean = df.drop_duplicates(subset=['body'], keep='first')

    removed_count = initial_count - len(df_clean)
    print(f"Removed {removed_count} duplicate rows")
    print(f"Remaining rows: {len(df_clean)}")

    return df_clean


def print_dataset_statistics(df):
    """
    Print statistics about the cleaned dataset.

    Args:
        df (pd.DataFrame): Dataframe to analyze
    """
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    print(f"Total samples: {len(df)}")
    print(f"\nPriority distribution:")
    priority_counts = df['priority'].value_counts().sort_index()
    for priority, count in priority_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {priority}: {count} ({percentage:.2f}%)")

    print(f"\nText length statistics (characters):")
    text_lengths = df['body'].str.len()
    print(f"  Mean: {text_lengths.mean():.2f}")
    print(f"  Median: {text_lengths.median():.2f}")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")
    print("="*50 + "\n")


def clean_dataset(input_path, output_path):
    """
    Main function to clean the dataset for BERT fine-tuning.

    Args:
        input_path (str): Path to raw CSV file
        output_path (str): Path to save cleaned CSV file
    """
    print("Starting data cleaning process...\n")

    # Load data
    df = load_data(input_path)

    # Select relevant columns
    df = select_relevant_columns(df)

    # Remove null values
    df = remove_null_values(df)

    # Normalize priority labels
    df = normalize_priority_labels(df)

    # Apply text cleaning
    df = apply_text_cleaning(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Print statistics
    print_dataset_statistics(df)

    # Save cleaned dataset
    print(f"Saving cleaned dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} rows to {output_path}")

    print("\nData cleaning completed successfully!")
    return df


if __name__ == "__main__":
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "data", "raw.csv")
    output_file = os.path.join(script_dir, "data", "clean.csv")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        exit(1)

    # Run cleaning
    clean_dataset(input_file, output_file)
