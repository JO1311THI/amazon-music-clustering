# Amazon Music Clustering

This project clusters songs and artists from Amazon Music using unsupervised machine learning. It provides tools for data preprocessing, feature engineering, clustering (KMeans, DBSCAN, Agglomerative), and visualization.

## Project Structure

- `app.py`: Streamlit web app for interactive clustering and visualization.
- `src/`
  - `clustering.py`: Clustering algorithms and feature preparation ([src/clustering.py](src/clustering.py)).
  - `data_utils.py`: Data loading, cleaning, feature selection, scaling.
  - `viz.py`: Cluster visualization functions.
- `data/raw/`: Raw CSV data files.
- `results/`: Saved models and processed data.
- `notebooks/`: Jupyter notebooks for EDA and preprocessing.

## Quickstart

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Run preprocessing**  
   Use the notebook [notebooks/eda_preprocessing.ipynb](notebooks/eda_preprocessing.ipynb) or run:
   ```python
   from src.data_utils import preprocess
   preprocess(
       path="data/raw/single_genre_artists.csv",
       save_clean="results/clean_data.csv",
       save_scaler="results/scaler.joblib"
   )
   ```

3. **Launch the app**  
   ```
   streamlit run app.py
   ```

## Main Features

- Data cleaning and feature engineering ([src/data_utils.py](src/data_utils.py))
- Clustering: KMeans, DBSCAN, Agglomerative ([src/clustering.py](src/clustering.py))
- Cluster visualization ([src/viz.py](src/viz.py))
- Interactive UI with Streamlit ([app.py](app.py))

## Data Columns

- Song features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`
- Metadata: `track_name`, `artist_name`, `track_id`, `release_date`, `followers`, `genres`, etc.

## References

- [`prepare_features`](src/clustering.py)
- [`run_kmeans`](src/clustering.py)
- [`plot_clusters`](src/viz.py)
- [`preprocess`](src/data_utils.py)

## License

MIT License