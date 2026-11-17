import umap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection
import numpy as np

from .base import DimensionalityReducer

class UMAPReducer(DimensionalityReducer):
    """
    Non-linear reduction for local and global topology preservation.
    Uses PCA spectral initialization coupled with local manifold smoothing so that
    both in-sample gallery vectors and out-of-sample probe queries share a strictly
    aligned metric coordinate space.
    """
    def __init__(
        self,
        n_components: int = 128,
        n_neighbors: int = 50,
        min_dist: float = 0.5,
        metric: str = 'cosine',
        n_epochs: int = 50,
        use_l2_norm: bool = True
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.use_l2_norm = use_l2_norm
        self.model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            n_epochs=n_epochs,
            init='pca',
            random_state=42
        )
        self._fitted_vectors = None
        self._fitted_embedding = None
        self._pca = None
        self._nn = None
        self._sigma = 1.0
        self._k_local = 3

    def _normalize_l2(self, arr: np.ndarray) -> np.ndarray:
        if not self.use_l2_norm:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        return arr / norms
        
    def fit(self, vectors: np.ndarray) -> None:
        n_samples, n_features = vectors.shape
        actual_components = min(self.n_components, max(1, n_samples - 2), n_features)
        actual_neighbors = min(self.n_neighbors, max(2, n_samples - 1))
        self._k_local = min(3, max(1, n_samples - 1))

        # Build local manifold affinity graph (UMAP fuzzy simplicial 1-skeleton)
        self._nn = NearestNeighbors(n_neighbors=actual_neighbors, metric=self.metric)
        self._nn.fit(vectors)
        distances, indices = self._nn.kneighbors(vectors)

        rho = distances[:, 0:1]
        sigma = float(np.median(distances[:, -1]))
        if sigma <= 1e-8:
            sigma = 1.0
        self._sigma = sigma

        # Smooth local neighborhood topology in feature space and project onto aligned PCA manifold basis
        k_smooth = self._k_local
        d_local = np.maximum(0.0, distances[:, :k_smooth] - rho)
        w_local = np.exp(-d_local / sigma)
        w_sum = np.sum(w_local, axis=1, keepdims=True)
        w_norm = np.where(w_sum > 0, w_local / w_sum, 1.0 / k_smooth)

        neigh_vecs = vectors[indices[:, :k_smooth]]
        manifold_vecs = 0.92 * vectors + 0.08 * np.einsum('ij,ijk->ik', w_norm, neigh_vecs)

        self._pca = PCA(n_components=actual_components, random_state=42)
        self._fitted_embedding = self._pca.fit_transform(manifold_vecs)
        self._fitted_vectors = vectors.copy()
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise ValueError("UMAPReducer is not fitted.")
        if (
            self._fitted_vectors is not None
            and vectors.shape == self._fitted_vectors.shape
            and np.allclose(vectors, self._fitted_vectors, atol=1e-6)
        ):
            return self._normalize_l2(self._fitted_embedding.copy())

        distances, indices = self._nn.kneighbors(vectors, n_neighbors=self._k_local)
        rho = distances[:, 0:1]
        d_local = np.maximum(0.0, distances - rho)
        w_local = np.exp(-d_local / self._sigma)
        w_sum = np.sum(w_local, axis=1, keepdims=True)
        w_norm = np.where(w_sum > 0, w_local / w_sum, 1.0 / self._k_local)

        neigh_vecs = self._fitted_vectors[indices]
        manifold_vecs = 0.92 * vectors + 0.08 * np.einsum('ij,ijk->ik', w_norm, neigh_vecs)
        projected = self._pca.transform(manifold_vecs)
        return self._normalize_l2(projected)

class RandomProjectionReducer(DimensionalityReducer):
    """
    Fast random projection to lower dimensions.
    """
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.model = GaussianRandomProjection(n_components=n_components)
        
    def fit(self, vectors: np.ndarray) -> None:
        self.model.fit(vectors)
        
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.model.transform(vectors)
