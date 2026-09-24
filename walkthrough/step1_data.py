"""
WALKTHROUGH STEP 1: The data.

This only READS the dataset files and prints them. It changes nothing.
"""
import pandas as pd

LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

# ---- 1. How big is each split? -------------------------------------------
print("=" * 60)
print("1. HOW MANY REVIEWS ARE IN EACH SPLIT")
print("=" * 60)
for name, df in [("train", train), ("validation", val), ("test", test)]:
    counts = df["label"].astype(int).value_counts().sort_index()
    parts = "  ".join(f"{LABELS[k]}: {v}" for k, v in counts.items())
    print(f"{name:<11} {len(df):>5} reviews  ->  {parts}")

# ---- 2. What does a review look like? -------------------------------------
print("\n" + "=" * 60)
print("2. ONE EXAMPLE REVIEW PER LABEL (from train)")
print("=" * 60)
for label in (0, 1, 2):
    row = train[train["label"] == label].iloc[0]
    print(f"\n[{LABELS[label]}]")
    print(f"  {row['review']}")

# ---- 3. What did cleaning change? -----------------------------------------
print("\n" + "=" * 60)
print("3. WHAT THE CLEANING STEP CHANGED (first 3 reviews it changed)")
print("=" * 60)
changed = train[train["review"] != train["clean_review"]].head(3)
for _, row in changed.iterrows():
    print(f"\n  before: {row['review']}")
    print(f"  after:  {row['clean_review']}")
