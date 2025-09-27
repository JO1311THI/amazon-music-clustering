import sys, os
sys.path.append(os.path.abspath(".."))  # ✅ Ensure project root is in PYTHONPATH

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import joblib

from src.viz import plot_clusters  # ✅ works now

def prepare_features(df, features=None):
    if features is None:
        features = [
            "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"
        ]
    X = df[features].values
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def run_kmeans(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model

def run_agglomerative(X, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels, model
