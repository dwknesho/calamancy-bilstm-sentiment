import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Collapses repeated letters ("sobraaaaa" -> "sobraa") and strips punctuation."""
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)      # 3+ repeats -> 2
    text = re.sub(r"[^a-zA-Z0-9\s']", '', text)     # keep letters/digits/apostrophes

    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
    """Adds a clean_review column."""
    df = df.copy()
    df["clean_review"] = df[text_col].apply(clean_text)
    return df


def stratified_split(df: pd.DataFrame, label_col: str = "label", val_size: float = 0.10, seed: int = 42):
    """Stratified train/val split on label_col."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df[label_col],
        random_state=seed
    )
    return train_df, val_df