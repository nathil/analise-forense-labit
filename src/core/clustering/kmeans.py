from sklearn.cluster import KMeans as SKKMeans
import numpy as np

from .base import Clusterer

class KMeansClusterer(Clusterer):
    """
    KMeans implementation.
    If the specified metric is 'cosine', automatically applies L2 normalization 
    to vectors prior to clustering so that Euclidean distance behaves 
    isomorphically to Cosine distance.
    """
    def __init__(self, n_clusters: int = 8, metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric.lower()
        self.model = SKKMeans(n_clusters=n_clusters, random_state=42)
        
    def _preprocess(self, vectors: np.ndarray) -> np.ndarray:
        if self.metric == 'cosine':
            # Apply L2 normalization
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1e-10
            return vectors / norms
        return vectors

    def fit(self, vectors: np.ndarray) -> None:
        processed_vectors = self._preprocess(vectors)
        self.model.fit(processed_vectors)
        
    def predict(self, vectors: np.ndarray) -> np.ndarray:
        processed_vectors = self._preprocess(vectors)
        return self.model.predict(processed_vectors)
        
    @property
    def cluster_centers(self) -> np.ndarray:
        return self.model.cluster_centers_

from sklearn.cluster import MiniBatchKMeans as SKMiniBatchKMeans

class MiniBatchKMeansClusterer(Clusterer):
    """
    MiniBatch KMeans implementation.
    """
    def __init__(self, n_clusters: int = 8, metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric.lower()
        self.model = SKMiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        
    def _preprocess(self, vectors: np.ndarray) -> np.ndarray:
        if self.metric in ['cosine', 'dot']:
            # Apply L2 normalization
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            return vectors / norms
        return vectors

    def fit(self, vectors: np.ndarray) -> None:
        processed_vectors = self._preprocess(vectors)
        self.model.fit(processed_vectors)
        
    def predict(self, vectors: np.ndarray) -> np.ndarray:
        processed_vectors = self._preprocess(vectors)
        return self.model.predict(processed_vectors)
        
    @property
    def cluster_centers(self) -> np.ndarray:
        return self.model.cluster_centers_
