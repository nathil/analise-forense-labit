from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np

from .base import DimensionalityReducer

class PCAReducer(DimensionalityReducer):
    """
    Linear projection to a lower dimension (e.g. 64, 128, 256) 
    maximizing explained variance.
    """
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.model = PCA(n_components=n_components)
        
    def fit(self, vectors: np.ndarray) -> None:
        self.model.fit(vectors)
        
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.model.transform(vectors)

class SVDReducer(DimensionalityReducer):
    """
    Truncated SVD. Ideal for sparse matrices or equivalent algebraic reductions.
    Does not center the data before computing the singular value decomposition.
    """
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components)
        
    def fit(self, vectors: np.ndarray) -> None:
        self.model.fit(vectors)
        
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.model.transform(vectors)
