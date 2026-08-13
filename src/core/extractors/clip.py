import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Union

from .base import FeatureExtractor

class CLIPExtractor(FeatureExtractor):
    """
    Extracts features using OpenAI's CLIP model (vision encoder).
    These vectors naturally align with cosine similarity metrics.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        img_pil = self._ensure_pil(image)
        
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        
        # Features are already L2 normalized by default in CLIP inference, but we ensure it's standard
        features = outputs.squeeze().cpu().numpy()
        
        return features
