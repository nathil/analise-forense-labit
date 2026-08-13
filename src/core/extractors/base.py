from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from typing import Union

class FeatureExtractor(ABC):
    """
    Base interface for all feature extractors.
    """
    @abstractmethod
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extracts a feature vector from an image.
        
        Args:
            image: A numpy array or PIL Image representing the tattoo.
            
        Returns:
            A 1D numpy array representing the feature vector.
        """
        pass
        
    def _ensure_pil(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Helper to convert numpy array to PIL Image if needed."""
        if isinstance(image, np.ndarray):
            # If it's a BGR image from cv2, convert to RGB
            if image.shape[-1] == 3:
                # Naive assumption it's RGB, if it was read by cv2 it might be BGR,
                # but typically we handle that before.
                pass
            return Image.fromarray(image)
        return image
