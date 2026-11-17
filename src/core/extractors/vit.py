import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from typing import Union

from .base import FeatureExtractor

class DINOv2Extractor(FeatureExtractor):
    """
    Extracts features using Facebook's DINOv2 ViT model.
    Captures advanced local and semantic properties of tattoo textures.
    """
    def __init__(self, model_name: str = "facebook/dinov2-base", use_l2_norm: bool = True, device: str = None):
        self.use_l2_norm = use_l2_norm
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        img_pil = self._ensure_pil(image)
        
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        # Pooler output or last hidden state CLS token
        features = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        if self.use_l2_norm:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
        
        return features

class ViTExtractor(FeatureExtractor):
    """
    Extracts features using a standard Vision Transformer (google/vit-base-patch16-224-in21k).
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", use_l2_norm: bool = True, device: str = None):
        self.use_l2_norm = use_l2_norm
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        img_pil = self._ensure_pil(image)
        
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        features = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        if self.use_l2_norm:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
        
        return features
