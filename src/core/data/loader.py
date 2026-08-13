import os
import re
from typing import List, Dict, Tuple
from pathlib import Path

class BivTattDatasetLoader:
    """
    Loads images and their corresponding labels from the BivTatt dataset.
    The dataset follows the nomenclature:
    - Original: label_number.jpg
    - Transformed: label_number_transformation.jpg
    
    It also finds the corresponding bounding box text files (x1 y1 x2 y2).
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.boxes_dir = self.root_dir / "bounding_boxes"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found at {self.images_dir}")

    def load_dataset(self) -> List[Dict]:
        """
        Scans the directory and builds a list of dictionaries containing
        metadata for each image.
        """
        dataset = []
        for file_path in self.images_dir.glob("*.jpg"):
            filename = file_path.stem
            parts = filename.split("_")
            
            # Extract label
            label = parts[0]
            
            # Check if it's an original image (gallery) or transformed (probe)
            is_original = len(parts) == 2
            transformation = parts[2] if not is_original else None
            
            # Check for bounding box file
            box_path = self.boxes_dir / f"{filename}.txt"
            has_box = box_path.exists()
            
            dataset.append({
                "image_path": str(file_path),
                "box_path": str(box_path) if has_box else None,
                "label": label,
                "is_original": is_original,
                "transformation": transformation,
                "filename": file_path.name
            })
            
        return dataset

class DatasetSplitter:
    """
    Splits the dataset into Gallery and Probe sets based on the forensic protocol.
    Gallery: Original images (clean).
    Probe: Transformed images.
    """
    @staticmethod
    def split(dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns a tuple of (Gallery, Probe).
        Ensures zero leakage between the sets based on the 'is_original' flag.
        """
        gallery = []
        probe = []
        
        for item in dataset:
            if item["is_original"]:
                gallery.append(item)
            else:
                probe.append(item)
                
        return gallery, probe
