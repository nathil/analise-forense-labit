import pytest
import numpy as np
from core.data.loader import DatasetSplitter
from core.data.preprocessor import BoundingBoxCrop

def test_bounding_box_valid_crop():
    # Create a dummy image of 100x100
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # x1, y1, x2, y2
    box = (10, 20, 50, 60)
    
    cropped = BoundingBoxCrop.crop_image(image, box)
    
    assert cropped.shape == (40, 40, 3), "Crop should match the box dimensions"

def test_bounding_box_out_of_bounds():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box goes beyond image bounds
    box = (50, 50, 150, 150)
    
    cropped = BoundingBoxCrop.crop_image(image, box)
    
    # Should clamp to image bounds: x2->100, y2->100
    # resulting in (50, 50)
    assert cropped.shape == (50, 50, 3), "Crop should clamp to image boundaries"

def test_bounding_box_invalid_coords():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # x1 > x2
    box = (60, 20, 50, 60)
    
    cropped = BoundingBoxCrop.crop_image(image, box)
    
    # Should return original image
    assert cropped.shape == (100, 100, 3), "Invalid crop should return original image"

def test_dataset_splitter_zero_leakage():
    # Create mock dataset
    dataset = [
        {"image_path": "a_1.jpg", "label": "a", "is_original": True},
        {"image_path": "a_2.jpg", "label": "a", "is_original": True},
        {"image_path": "a_1_trans1.jpg", "label": "a", "is_original": False},
        {"image_path": "b_1.jpg", "label": "b", "is_original": True},
        {"image_path": "b_1_trans1.jpg", "label": "b", "is_original": False},
    ]
    
    gallery, probe = DatasetSplitter.split(dataset)
    
    assert len(gallery) == 3, "Gallery should have 3 items"
    assert len(probe) == 2, "Probe should have 2 items"
    
    # Ensure no item in gallery has is_original == False
    for item in gallery:
        assert item["is_original"] is True
        
    # Ensure no item in probe has is_original == True
    for item in probe:
        assert item["is_original"] is False
