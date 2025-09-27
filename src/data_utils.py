import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    """
    Clean dataset by dropping unnecessary columns and duplicates.
    """
    if drop_cols is None:
        drop_cols = ["track_name", "artist_name", "track_id"]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select numerical features for clustering and apply log transform on duration.
    """
    features = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
    ]

    X = df[features].copy()

    if "duration_ms" in X.columns:
        X["duration_ms"] = np.log1p(X["duration_ms"])  # log transform

    return X


def scale_features(X: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler and optionally save the scaler.
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    if save_path:
        joblib.dump(scaler, save_path)

    return X_scaled


def preprocess(path: str, save_clean: str = None, save_scaler: str = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Load dataset
    2. Clean dataset
    3. Select features
    4. Scale features
    5. Save cleaned dataset & scaler (optional)
    """
    df = load_dataset(path)
    df = clean_dataset(df)
    X = select_features(df)
    X_scaled = scale_features(X, save_path=save_scaler)

    # attach scaled features back to df
    df_clean = df.copy()
    df_clean[X_scaled.columns] = X_scaled

    if save_clean:
        df_clean.to_csv(save_clean, index=False)

    return df_clean
