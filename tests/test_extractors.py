import pytest
import numpy as np
from PIL import Image

# Import extractors (we mock them or run on CPU lightly to test shapes if possible)
from core.extractors.cnn import ResNet50Extractor

# Note: DINOv2 and CLIP might download weights if we instantiate them, 
# so we might want to mock the AutoModel or just test ResNet which is relatively small.
# For demonstration in Phase 2 tests, we will test ResNet.

def test_resnet50_extractor_shape():
    extractor = ResNet50Extractor(use_l2_norm=True)
    
    # Create dummy image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    vector = extractor.extract(image)
    
    assert vector.shape == (2048,), "ResNet50 feature vector should be 2048-dimensional"
    
    # Check L2 normalization
    norm = np.linalg.norm(vector)
    assert np.isclose(norm, 1.0, atol=1e-5), "Vector should be L2 normalized"

def test_ensure_pil_conversion():
    extractor = ResNet50Extractor()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pil_img = extractor._ensure_pil(image)
    assert isinstance(pil_img, Image.Image), "Should convert numpy to PIL"
