from abc import ABC, abstractmethod
import numpy as np

class Clusterer(ABC):
    """
    Base interface for all clustering algorithms.
    """
    @abstractmethod
    def fit(self, vectors: np.ndarray) -> None:
        """
        Fits the clustering model to the vectors.
        """
        pass
        
    @abstractmethod
    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        Predicts the closest cluster each sample in vectors belongs to.
        Returns an array of cluster labels.
        """
        pass
        
    @property
    @abstractmethod
    def cluster_centers(self) -> np.ndarray:
        """
        Returns the centroids or representative vectors of the clusters.
        """
        pass
