from sklearn.cluster import HDBSCAN, MiniBatchKMeans
import numpy as np

from .base import Clusterer

class HDBSCANClusterer(Clusterer):
    """
    HDBSCAN implementation with L2-normalization support for cosine/dot metrics
    and automatic fallback if high-dimensional density estimation marks all points as noise.
    """
    def __init__(self, min_cluster_size: int = 3, metric: str = 'euclidean', copy: bool = True):
        self.min_cluster_size = max(2, min_cluster_size)
        self.metric = metric.lower()
        self.copy = copy
        # On L2-normalized vectors, Euclidean distance is monotonically equivalent to Cosine/Dot
        sk_metric = 'euclidean' if self.metric in ['cosine', 'dot', 'euclidean'] else self.metric
        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
            metric=sk_metric,
            cluster_selection_method='eom',
            copy=copy
        )
        self._cluster_centers = None
        self.labels_ = None

    def _preprocess(self, vectors: np.ndarray) -> np.ndarray:
        if self.metric in ['cosine', 'dot']:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            return vectors / norms
        return vectors
        
    def fit(self, vectors: np.ndarray) -> None:
        proc_vectors = self._preprocess(vectors)
        if len(proc_vectors) < self.min_cluster_size:
            self.labels_ = np.zeros(len(proc_vectors), dtype=int)
            self._cluster_centers = self._preprocess(np.mean(proc_vectors, axis=0, keepdims=True))
            return

        self.model.fit(proc_vectors)
        self.labels_ = self.model.labels_
        
        unique_labels = sorted(l for l in set(self.labels_) if l != -1)
        
        if (len(unique_labels) < 2 or len(unique_labels) > 10) and len(proc_vectors) >= 4:
            # Fallback/consolidate to balanced MiniBatchKMeans centroids if HDBSCAN collapses into noise
            # or fragments the gallery into dozens of tiny micro-clusters
            k_fallback = min(8, max(2, len(proc_vectors) // 8))
            km = MiniBatchKMeans(n_clusters=k_fallback, random_state=42)
            self.labels_ = km.fit_predict(proc_vectors)
            self._cluster_centers = self._preprocess(km.cluster_centers_)
        elif not unique_labels:
            self._cluster_centers = self._preprocess(np.mean(proc_vectors, axis=0, keepdims=True))
        else:
            centers = []
            for lbl in unique_labels:
                cluster_pts = proc_vectors[self.labels_ == lbl]
                if len(cluster_pts) > 0:
                    centers.append(np.mean(cluster_pts, axis=0))
            self._cluster_centers = self._preprocess(np.array(centers))

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        Approximate predict by finding nearest centroid.
        """
        from sklearn.neighbors import NearestNeighbors
        if self._cluster_centers is None:
            raise ValueError("Model is not fitted yet.")
            
        proc_vectors = self._preprocess(vectors)
        nn_metric = 'cosine' if self.metric in ['cosine', 'dot'] else self.metric
        nn = NearestNeighbors(n_neighbors=1, metric=nn_metric)
        nn.fit(self._cluster_centers)
        
        _, indices = nn.kneighbors(proc_vectors)
        return indices.flatten()
        
    @property
    def cluster_centers(self) -> np.ndarray:
        return self._cluster_centers
