import pytest
import numpy as np
import cv2
from core.data.loader import DatasetSplitter, YOLOv5WeaponDatasetLoader
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


def test_yolo_bounding_box_parsing_and_crop(tmp_path):
    # Image of height=100, width=200
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    box_file = tmp_path / "sample.txt"
    # Two YOLO boxes: first is small (0.1 x 0.1), second is larger (cx=0.5, cy=0.5, w=0.4, h=0.6)
    box_file.write_text("0 0.2 0.2 0.1 0.1\n1 0.5 0.5 0.4 0.6\n", encoding="utf-8")

    box = BoundingBoxCrop.parse_box_file(str(box_file), image.shape[:2])
    # Expected for cx=0.5, cy=0.5, w=0.4, h=0.6 on (H=100, W=200):
    # x1 = (0.5 - 0.2)*200 = 60, x2 = (0.5 + 0.2)*200 = 140 -> width 80
    # y1 = (0.5 - 0.3)*100 = 20, y2 = (0.5 + 0.3)*100 = 80  -> height 60
    assert box == (60, 20, 140, 80)
    cropped = BoundingBoxCrop.crop_image(image, box)
    assert cropped.shape == (60, 80, 3)


def test_yolov5_weapon_dataset_loader(tmp_path):
    # Create mock YOLOv5 structure (train/images, train/labels, valid/images, valid/labels)
    train_img = tmp_path / "train" / "images"
    train_lbl = tmp_path / "train" / "labels"
    val_img = tmp_path / "valid" / "images"
    val_lbl = tmp_path / "valid" / "labels"
    for p in (train_img, train_lbl, val_img, val_lbl):
        p.mkdir(parents=True)

    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['knife', 'handgun']\n", encoding="utf-8")

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(6):
        # Class 0: knife
        img_k = train_img / f"k_{i}.jpg"
        cv2.imwrite(str(img_k), dummy_img)
        (train_lbl / f"k_{i}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

        # Class 1: handgun
        img_h = val_img / f"h_{i}.jpg"
        cv2.imwrite(str(img_h), dummy_img)
        (val_lbl / f"h_{i}.txt").write_text("1 0.5 0.5 0.6 0.4\n", encoding="utf-8")

    loader = YOLOv5WeaponDatasetLoader(str(tmp_path))
    dataset = loader.load_dataset()

    assert len(dataset) == 12
    labels = {item["label"] for item in dataset}
    assert labels == {"knife", "handgun"}

    gallery, probe = DatasetSplitter.split(dataset)
    assert len(gallery) > 0 and len(probe) > 0
    assert {g["label"] for g in gallery} == {"knife", "handgun"}
    assert {p["label"] for p in probe} == {"knife", "handgun"}
