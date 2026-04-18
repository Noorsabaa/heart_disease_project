"""src/models/unsupervised.py — K-Means clustering and PCA."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.config import RANDOM_STATE


def build_kmeans(n_clusters=2):
    """K-Means with k=2 (disease vs no-disease). Run on PCA-reduced space."""
    return KMeans(n_clusters=n_clusters, n_init=20, max_iter=500, random_state=RANDOM_STATE)


def build_pca(n_components=2):
    """PCA to 2D for cluster visualisation and dimensionality reduction."""
    return PCA(n_components=n_components, random_state=RANDOM_STATE)
