import umap
from sklearn.random_projection import GaussianRandomProjection
import numpy as np

from .base import DimensionalityReducer

class UMAPReducer(DimensionalityReducer):
    """
    Non-linear reduction for local topology preservation.
    """
    def __init__(self, n_components: int = 128, n_neighbors: int = 15, metric: str = 'cosine'):
        self.n_components = n_components
        self.model = umap.UMAP(n_neighbors=n_neighbors, 
                               n_components=n_components, 
                               metric=metric)
        
    def fit(self, vectors: np.ndarray) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_neighbors is larger than the dataset size")
            self.model.fit(vectors)
        
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.model.transform(vectors)

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
