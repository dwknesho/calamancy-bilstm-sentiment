import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Cleans a single review string per Ch3-C.2:
    - Collapses characters repeated more than 2 times in a row
      (e.g. "sobraaaaa" -> "sobraa")
    - Removes special/non-alphanumeric characters, but keeps apostrophes
    """
    # Collapse 3+ repeated characters down to 2
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Keep letters, numbers, spaces, and apostrophes only
    text = re.sub(r"[^a-zA-Z0-9\s']", '', text)

    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "review") -> pd.DataFrame:
    """
    Applies clean_text to every row of the given column.
    Returns a new dataframe with an added 'clean_review' column.
    """
    df = df.copy()
    df["clean_review"] = df[text_col].apply(clean_text)
    return df


def stratified_split(df: pd.DataFrame, label_col: str = "label", val_size: float = 0.10, seed: int = 42):
    """
    Splits df into train/val using stratified sampling on label_col.
    Returns (train_df, val_df).
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df[label_col],
        random_state=seed
    )
    return train_df, val_df