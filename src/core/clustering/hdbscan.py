from sklearn.cluster import HDBSCAN
import numpy as np

from .base import Clusterer

class HDBSCANClusterer(Clusterer):
    """
    HDBSCAN implementation.
    Note: HDBSCAN does not have standard cluster centers like KMeans,
    so we calculate the mean of the points in each cluster as the centroid.
    """
    def __init__(self, min_cluster_size: int = 5, metric: str = 'euclidean'):
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.model = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
        self._cluster_centers = None
        self.labels_ = None
        
    def fit(self, vectors: np.ndarray) -> None:
        self.model.fit(vectors)
        self.labels_ = self.model.labels_
        
        # Calculate centroids
        unique_labels = set(self.labels_)
        
        # HDBSCAN uses -1 for noise. We can group noise or ignore it.
        # Here we include it as a separate cluster or just ignore it for centers.
        valid_labels = [l for l in unique_labels if l != -1]
        
        if not valid_labels:
            # If everything is noise, just use one center at the mean
            self._cluster_centers = np.mean(vectors, axis=0, keepdims=True)
        else:
            n_clusters = max(valid_labels) + 1
            self._cluster_centers = np.zeros((n_clusters, vectors.shape[1]))
            for i in valid_labels:
                cluster_pts = vectors[self.labels_ == i]
                if len(cluster_pts) > 0:
                    self._cluster_centers[i] = np.mean(cluster_pts, axis=0)

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        Approximate predict by finding nearest centroid.
        """
        from sklearn.neighbors import NearestNeighbors
        if self._cluster_centers is None:
            raise ValueError("Model is not fitted yet.")
            
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        nn.fit(self._cluster_centers)
        
        _, indices = nn.kneighbors(vectors)
        return indices.flatten()
        
    @property
    def cluster_centers(self) -> np.ndarray:
        return self._cluster_centers
