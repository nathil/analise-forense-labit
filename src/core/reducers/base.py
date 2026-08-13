from abc import ABC, abstractmethod
import numpy as np

class DimensionalityReducer(ABC):
    """
    Base interface for all dimensionality reducers.
    """
    @abstractmethod
    def fit(self, vectors: np.ndarray) -> None:
        """
        Fits the dimensionality reduction model on the gallery vectors.
        
        Args:
            vectors: 2D numpy array of shape (N, D) where N is number of samples
                     and D is original dimensions.
        """
        pass
        
    @abstractmethod
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Transforms vectors into lower dimensional space.
        
        Args:
            vectors: 2D numpy array of shape (M, D)
            
        Returns:
            2D numpy array of shape (M, d) where d < D.
        """
        pass
