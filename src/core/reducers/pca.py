from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np

from .base import DimensionalityReducer

class PCAReducer(DimensionalityReducer):
    """
    Linear projection to a lower dimension (e.g. 64, 128, 256) 
    maximizing explained variance, with optional L2 normalization for
    rotation/illumination invariance in Euclidean distance retrieval.
    """
    def __init__(self, n_components: int = 128, use_l2_norm: bool = True):
        self.n_components = n_components
        self.use_l2_norm = use_l2_norm
        self.model = PCA(n_components=n_components)
        
    def fit(self, vectors: np.ndarray) -> None:
        self.model.fit(vectors)
        
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        projected = self.model.transform(vectors)
        if self.use_l2_norm:
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            projected = projected / norms
        return projected

class SVDReducer(DimensionalityReducer):
    """
    Truncated SVD. Ideal for sparse matrices or equivalent algebraic reductions.
    Does not center the data before computing the singular value decomposition.
    """
    def __init__(self, n_components: int = 128, use_l2_norm: bool = True):
        self.n_components = n_components
        self.use_l2_norm = use_l2_norm
        self.model = TruncatedSVD(n_components=n_components)
        
    def fit(self, vectors: np.ndarray) -> None:
        self.model.fit(vectors)
        
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        projected = self.model.transform(vectors)
        if self.use_l2_norm:
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            projected = projected / norms
        return projected
