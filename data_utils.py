"""
data_utils.py - Member 1 (Joseph Moarbes)
Handles dataset loading, cleaning, info display, and visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(filepath="spam.csv"):
    """Load and clean the SMS Spam Collection dataset."""
    df = pd.read_csv(filepath, encoding="latin-1")

    # Keep only relevant columns (dataset sometimes has extra unnamed cols)
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]

    # Fix common column name typo 'lable' -> 'label'
    df.columns = [col if col != "lable" else "label" for col in df.columns]

    # Drop any rows with missing values
    df.dropna(subset=["label", "message"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def show_dataset_info(df):
    """Print basic dataset info and label counts."""
    print("\n--- Dataset Info ---")
    print(f"Total messages: {len(df)}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nLabel counts:")
    counts = df["label"].value_counts()
    for label, count in counts.items():
        print(f"  {label.upper()}: {count}")
    print("--------------------\n")


def visualize_label_counts(df):
    """Generate a bar chart of spam vs ham message counts."""
    counts = df["label"].value_counts()
    colors = ["#e74c3c" if label == "spam" else "#2ecc71" for label in counts.index]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index.str.upper(), counts.values, color=colors, width=0.4)

    # Add count labels on top of bars
    for bar, count in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            str(count),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.title("Spam vs Ham Message Counts", fontsize=14, fontweight="bold")
    plt.xlabel("Message Type")
    plt.ylabel("Number of Messages")
    plt.ylim(0, counts.max() + 300)
    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=150)
    plt.show()
    print("Chart saved as label_distribution.png")
