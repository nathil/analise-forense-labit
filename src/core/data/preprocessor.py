import cv2
import numpy as np
from typing import Tuple

class BoundingBoxCrop:
    """
    Handles cropping of the tattoo region strictly based on provided coordinates.
    """
    @staticmethod
    def parse_box_file(box_path: str) -> Tuple[int, int, int, int]:
        """
        Reads a text file containing 'x1 y1 x2 y2' and returns the tuple.
        Returns (0, 0, 0, 0) if reading fails.
        """
        try:
            with open(box_path, 'r') as f:
                content = f.read().strip()
                # Split by whitespace and convert to int
                coords = [int(float(x)) for x in content.split()]
                if len(coords) >= 4:
                    return coords[0], coords[1], coords[2], coords[3]
        except Exception as e:
            print(f"Failed to read box file {box_path}: {e}")
            
        return (0, 0, 0, 0)

    @staticmethod
    def crop_image(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crops the numpy array image using the box coordinates (x1, y1, x2, y2).
        If the box is invalid or (0,0,0,0), returns the original image.
        """
        x1, y1, x2, y2 = box
        
        # Validate coordinates against image bounds
        h, w = image.shape[:2]
        
        # Ensure coordinates are within valid range and form a valid rectangle
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        if x1 >= x2 or y1 >= y2:
            # Invalid box, return original
            return image
            
        return image[y1:y2, x1:x2]
