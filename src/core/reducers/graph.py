import numpy as np
from sklearn.manifold import SpectralEmbedding

from .base import DimensionalityReducer

class KNNGraphReducer(DimensionalityReducer):
    """
    Dimensionality reduction using a K-Nearest Neighbors Graph (Spectral Embedding).
    Builds a KNN graph and projects it to a lower dimensional space using Laplacian Eigenmaps.
    """
    def __init__(self, n_components: int = 128, n_neighbors: int = 15):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.model = SpectralEmbedding(
            n_components=self.n_components,
            affinity='nearest_neighbors',
            n_neighbors=self.n_neighbors
        )

    def fit(self, features: np.ndarray) -> None:
        n_samples = features.shape[0]
        actual_components = min(self.n_components, n_samples // 2)
        if actual_components < 1:
            actual_components = 1
            
        actual_neighbors = min(self.n_neighbors, max(1, n_samples - 1))
            
        self.model.set_params(n_components=actual_components, n_neighbors=actual_neighbors)
        reduced = self.model.fit_transform(features)
        
        # Train a regressor to approximate the transform for out-of-sample data
        from sklearn.neighbors import KNeighborsRegressor
        self._regressor = KNeighborsRegressor(n_neighbors=actual_neighbors)
        self._regressor.fit(features, reduced)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Approximates the Spectral Embedding for out-of-sample data using KNN Regression.
        """
        if not hasattr(self, '_regressor'):
            raise ValueError("KNNGraphReducer is not fitted.")
        return self._regressor.predict(features)

