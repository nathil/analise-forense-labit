from .base import DimensionalityReducer
import numpy as np

class PassThroughReducer(DimensionalityReducer):
    """
    A reducer that does nothing. Useful when the pipeline doesn't require 
    dimensionality reduction (e.g. relying directly on graph-based clustering).
    """
    def fit(self, vectors: np.ndarray) -> None:
        pass
        
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return vectors
