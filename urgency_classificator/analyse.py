"""
Dataset analysis script for urgency classification.
Provides comprehensive statistics and visualizations of the cleaned dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import re


def load_clean_data(file_path):
    """
    Load the cleaned CSV dataset.

    Args:
        file_path (str): Path to the cleaned CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print(f"Loading cleaned data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples\n")
    return df


def basic_statistics(df):
    """
    Display basic statistics about the dataset.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("="*70)
    print("BASIC DATASET STATISTICS")
    print("="*70)

    print(f"\nTotal number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Features: {list(df.columns)}")

    print("\n" + "-"*70)
    print("Missing Values:")
    print("-"*70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found")
    else:
        print(missing)

    print("\n" + "-"*70)
    print("Data Types:")
    print("-"*70)
    print(df.dtypes)
    print()


def priority_distribution_analysis(df):
    """
    Analyze the distribution of priority labels.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("="*70)
    print("PRIORITY DISTRIBUTION ANALYSIS")
    print("="*70)

    priority_counts = df['priority'].value_counts().sort_index()
    priority_percentages = (priority_counts / len(df) * 100).round(2)

    print("\nPriority Label Counts:")
    print("-"*70)
    for priority in ['low', 'medium', 'high']:
        if priority in priority_counts.index:
            count = priority_counts[priority]
            percentage = priority_percentages[priority]
            bar = '█' * int(percentage / 2)
            print(f"{priority.upper():8} | {count:5d} ({percentage:5.2f}%) | {bar}")

    print("\n" + "-"*70)
    print("Class Balance:")
    print("-"*70)

    # Calculate class imbalance ratio
    max_count = priority_counts.max()
    min_count = priority_counts.min()
    imbalance_ratio = max_count / min_count

    print(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")

    if imbalance_ratio < 1.5:
        print("✓ Dataset is well-balanced")
    elif imbalance_ratio < 3:
        print("⚠ Dataset has moderate imbalance")
    else:
        print("⚠ Dataset has significant imbalance - consider using class weights or resampling")
    print()


def text_length_analysis(df):
    """
    Analyze text length statistics.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("="*70)
    print("TEXT LENGTH ANALYSIS")
    print("="*70)

    df['char_count'] = df['body'].str.len()
    df['word_count'] = df['body'].str.split().str.len()
    df['sentence_count'] = df['body'].str.count(r'[.!?]+') + 1

    print("\nCharacter Count Statistics:")
    print("-"*70)
    print(f"Mean:   {df['char_count'].mean():.2f}")
    print(f"Median: {df['char_count'].median():.2f}")
    print(f"Std:    {df['char_count'].std():.2f}")
    print(f"Min:    {df['char_count'].min()}")
    print(f"Max:    {df['char_count'].max()}")

    print("\nWord Count Statistics:")
    print("-"*70)
    print(f"Mean:   {df['word_count'].mean():.2f}")
    print(f"Median: {df['word_count'].median():.2f}")
    print(f"Std:    {df['word_count'].std():.2f}")
    print(f"Min:    {df['word_count'].min()}")
    print(f"Max:    {df['word_count'].max()}")

    print("\nSentence Count Statistics:")
    print("-"*70)
    print(f"Mean:   {df['sentence_count'].mean():.2f}")
    print(f"Median: {df['sentence_count'].median():.2f}")
    print(f"Std:    {df['sentence_count'].std():.2f}")

    # BERT token limit warning
    print("\n" + "-"*70)
    print("BERT Token Considerations:")
    print("-"*70)
    # Rough estimate: ~1.3 tokens per word for multilingual text
    estimated_tokens = df['word_count'] * 1.3
    print(f"Estimated avg tokens: {estimated_tokens.mean():.2f}")
    print(f"Estimated max tokens: {estimated_tokens.max():.2f}")

    over_512 = (estimated_tokens > 512).sum()
    if over_512 > 0:
        print(
            f"⚠ {over_512} samples ({over_512/len(df)*100:.2f}%) may exceed BERT's 512 token limit")
        print("  Consider truncation or splitting long texts")
    else:
        print("✓ All samples should fit within BERT's 512 token limit")

    print()


def text_length_by_priority(df):
    """
    Analyze text length by priority class.

    Args:
        df (pd.DataFrame): Input dataframe with char_count column
    """
    print("="*70)
    print("TEXT LENGTH BY PRIORITY CLASS")
    print("="*70)

    print("\nAverage Character Count by Priority:")
    print("-"*70)
    for priority in ['low', 'medium', 'high']:
        if priority in df['priority'].values:
            avg_chars = df[df['priority'] == priority]['char_count'].mean()
            print(f"{priority.upper():8} | {avg_chars:.2f} characters")

    print("\nAverage Word Count by Priority:")
    print("-"*70)
    for priority in ['low', 'medium', 'high']:
        if priority in df['priority'].values:
            avg_words = df[df['priority'] == priority]['word_count'].mean()
            print(f"{priority.upper():8} | {avg_words:.2f} words")
    print()


def language_analysis(df):
    """
    Detect and analyze languages in the dataset.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("="*70)
    print("LANGUAGE ANALYSIS")
    print("="*70)

    def detect_language(text):
        """Simple language detection based on common words."""
        text_lower = text.lower()

        # German indicators
        german_words = ['der', 'die', 'das', 'und', 'ich', 'sie', 'wir', 'ist', 'sind',
                        'für', 'mit', 'auf', 'von', 'zu', 'dass', 'nicht', 'werden']

        # English indicators
        english_words = ['the', 'and', 'for', 'are', 'with', 'this', 'that', 'have',
                         'from', 'they', 'would', 'there', 'their', 'what', 'about']

        german_count = sum(
            1 for word in german_words if f' {word} ' in f' {text_lower} ')
        english_count = sum(
            1 for word in english_words if f' {word} ' in f' {text_lower} ')

        if german_count > english_count:
            return 'German'
        elif english_count > german_count:
            return 'English'
        else:
            return 'Unknown'

    df['detected_language'] = df['body'].apply(detect_language)

    print("\nDetected Languages:")
    print("-"*70)
    lang_counts = df['detected_language'].value_counts()
    for lang, count in lang_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{lang:10} | {count:5d} ({percentage:5.2f}%)")

    print("\n" + "-"*70)
    print("Language Distribution by Priority:")
    print("-"*70)
    cross_tab = pd.crosstab(df['detected_language'], df['priority'])
    print(cross_tab)
    print()


def vocabulary_analysis(df, top_n=20):
    """
    Analyze vocabulary and common words.

    Args:
        df (pd.DataFrame): Input dataframe
        top_n (int): Number of top words to display
    """
    print("="*70)
    print(f"VOCABULARY ANALYSIS (Top {top_n} words)")
    print("="*70)

    # Combine all text
    all_text = ' '.join(df['body'].values).lower()

    # Remove common punctuation but keep words
    all_text = re.sub(r'[^\w\s]', ' ', all_text)
    words = all_text.split()

    # Common stopwords (basic list)
    stopwords = {
        'the', 'and', 'for', 'are', 'with', 'this', 'that', 'have', 'from',
        'they', 'would', 'there', 'their', 'what', 'about', 'which', 'when',
        'than', 'them', 'these', 'could', 'other', 'into', 'has', 'more',
        'her', 'his', 'she', 'was', 'been', 'were', 'said', 'did', 'you',
        'der', 'die', 'das', 'und', 'ist', 'den', 'des', 'dem', 'ein', 'eine',
        'ich', 'sie', 'wir', 'für', 'mit', 'auf', 'von', 'nicht', 'den', 'sich'
    }

    # Filter out stopwords and short words
    filtered_words = [w for w in words if len(w) > 2 and w not in stopwords]

    word_freq = Counter(filtered_words)

    print(f"\nTotal unique words: {len(set(words))}")
    print(
        f"Total unique words (without stopwords): {len(set(filtered_words))}")
    print(f"\nTop {top_n} Most Common Words:")
    print("-"*70)

    for i, (word, count) in enumerate(word_freq.most_common(top_n), 1):
        print(f"{i:2d}. {word:20} | {count:5d} occurrences")
    print()


def sample_examples(df, n_per_class=3):
    """
    Display sample examples from each priority class.

    Args:
        df (pd.DataFrame): Input dataframe
        n_per_class (int): Number of samples per class
    """
    print("="*70)
    print(f"SAMPLE EXAMPLES ({n_per_class} per class)")
    print("="*70)

    for priority in ['low', 'medium', 'high']:
        if priority in df['priority'].values:
            print(f"\n{priority.upper()} Priority Examples:")
            print("-"*70)

            samples = df[df['priority'] == priority].sample(
                min(n_per_class, len(df[df['priority'] == priority])))

            for idx, (_, row) in enumerate(samples.iterrows(), 1):
                text = row['body']
                # Truncate long texts
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"\n{idx}. {text}")
    print()


def create_visualizations(df, output_dir):
    """
    Create and save visualization plots.

    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save plots
    """
    print("="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Priority Distribution Pie Chart
    print("\n1. Creating priority distribution chart...")
    fig, ax = plt.subplots(figsize=(10, 7))
    priority_counts = df['priority'].value_counts()
    colors = ['#ff6b6b', '#feca57', '#48dbfb']  # Red, Yellow, Blue
    ax.pie(priority_counts.values, labels=priority_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Priority Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'priority_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Character Count Distribution
    print("2. Creating character count distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Overall distribution
    axes[0].hist(df['char_count'], bins=50, color='steelblue',
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Character Count', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Character Count Distribution',
                      fontsize=14, fontweight='bold')
    axes[0].axvline(df['char_count'].mean(), color='red',
                    linestyle='--', linewidth=2, label='Mean')
    axes[0].legend()

    # By priority
    for priority in ['low', 'medium', 'high']:
        if priority in df['priority'].values:
            data = df[df['priority'] == priority]['char_count']
            axes[1].hist(data, bins=30, alpha=0.5,
                         label=priority.upper(), edgecolor='black')

    axes[1].set_xlabel('Character Count', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Character Count by Priority',
                      fontsize=14, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'character_count_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Word Count Distribution
    print("3. Creating word count distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].hist(df['word_count'], bins=50, color='forestgreen',
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Word Count', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Word Count Distribution',
                      fontsize=14, fontweight='bold')
    axes[0].axvline(df['word_count'].mean(), color='red',
                    linestyle='--', linewidth=2, label='Mean')
    axes[0].legend()

    # Box plot by priority
    priority_order = ['low', 'medium', 'high']
    data_to_plot = [df[df['priority'] == p]['word_count'].values
                    for p in priority_order if p in df['priority'].values]
    labels_to_plot = [p.upper()
                      for p in priority_order if p in df['priority'].values]

    axes[1].boxplot(data_to_plot, labels=labels_to_plot)
    axes[1].set_ylabel('Word Count', fontsize=12)
    axes[1].set_xlabel('Priority', fontsize=12)
    axes[1].set_title('Word Count by Priority', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'word_count_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Language Distribution
    if 'detected_language' in df.columns:
        print("4. Creating language distribution chart...")
        fig, ax = plt.subplots(figsize=(10, 6))
        lang_counts = df['detected_language'].value_counts()
        bars = ax.bar(lang_counts.index, lang_counts.values,
                      color=['#3498db', '#e74c3c', '#95a5a6'])
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Language Distribution', fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, 'language_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n✓ All visualizations saved to: {output_dir}")
    print()


def generate_summary_report(df, output_file):
    """
    Generate a text summary report.

    Args:
        df (pd.DataFrame): Input dataframe
        output_file (str): Path to save the report
    """
    print("="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("URGENCY CLASSIFICATION DATASET ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total Samples: {len(df)}\n")
        f.write(
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 70 + "\n")
        f.write("PRIORITY DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        priority_counts = df['priority'].value_counts().sort_index()
        for priority, count in priority_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{priority.upper()}: {count} ({percentage:.2f}%)\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("TEXT LENGTH STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Character Count - Mean: {df['char_count'].mean():.2f}, "
                f"Median: {df['char_count'].median():.2f}, "
                f"Std: {df['char_count'].std():.2f}\n")
        f.write(f"Word Count - Mean: {df['word_count'].mean():.2f}, "
                f"Median: {df['word_count'].median():.2f}, "
                f"Std: {df['word_count'].std():.2f}\n")

        if 'detected_language' in df.columns:
            f.write("\n" + "-" * 70 + "\n")
            f.write("LANGUAGE DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            lang_counts = df['detected_language'].value_counts()
            for lang, count in lang_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{lang}: {count} ({percentage:.2f}%)\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("RECOMMENDATIONS FOR BERT FINE-TUNING\n")
        f.write("-" * 70 + "\n")

        # Class balance recommendation
        max_count = priority_counts.max()
        min_count = priority_counts.min()
        imbalance_ratio = max_count / min_count

        if imbalance_ratio > 2:
            f.write("⚠ Consider using class weights to handle class imbalance\n")

        # Token length recommendation
        estimated_max_tokens = df['word_count'].max() * 1.3
        if estimated_max_tokens > 512:
            f.write(
                "⚠ Some texts may exceed BERT's 512 token limit - implement truncation\n")

        # Language recommendation
        if 'detected_language' in df.columns and len(df['detected_language'].unique()) > 1:
            f.write(
                "ℹ Dataset contains multiple languages - consider using multilingual BERT\n")

        f.write("\n✓ Dataset is ready for BERT fine-tuning\n")

    print(f"✓ Summary report saved to: {output_file}\n")


def analyze_dataset(input_file):
    """
    Main analysis function.

    Args:
        input_file (str): Path to the cleaned CSV file
    """
    print("\n" + "="*70)
    print("URGENCY CLASSIFICATION DATASET ANALYSIS")
    print("="*70 + "\n")

    # Load data
    df = load_clean_data(input_file)

    # Run analyses
    basic_statistics(df)
    priority_distribution_analysis(df)
    text_length_analysis(df)
    text_length_by_priority(df)
    language_analysis(df)
    vocabulary_analysis(df, top_n=20)
    sample_examples(df, n_per_class=3)

    # Create visualizations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(script_dir, 'visualizations')
    create_visualizations(df, viz_dir)

    # Generate summary report
    report_file = os.path.join(script_dir, 'reports/analysis_report.txt')
    generate_summary_report(df, report_file)

    print("="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n✓ Visualizations saved to: {viz_dir}")
    print(f"✓ Summary report saved to: {report_file}\n")


if __name__ == "__main__":
    # Define path to cleaned data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_file = os.path.join(script_dir, "data", "clean.csv")

    # Check if file exists
    if not os.path.exists(clean_file):
        print(f"Error: Cleaned data file not found at {clean_file}")
        print("Please run clean_data.py first to generate the cleaned dataset.")
        exit(1)

    # Run analysis
    analyze_dataset(clean_file)
