from .base import FeatureExtractor
from .cnn import ResNet50Extractor
from .vit import DINOv2Extractor
from .clip import CLIPExtractor

__all__ = ["FeatureExtractor", "ResNet50Extractor", "DINOv2Extractor", "CLIPExtractor"]
