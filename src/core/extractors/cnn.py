import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Union

from .base import FeatureExtractor

class ResNet50Extractor(FeatureExtractor):
    """
    Extracts features using a pre-trained ResNet50 model.
    Removes the final classification layer to return the 2048-d feature vector.
    """
    def __init__(self, use_l2_norm: bool = True, device: str = None):
        self.use_l2_norm = use_l2_norm
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load pre-trained model and remove fully connected layer
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # We replace the fc layer with an Identity layer
        model.fc = torch.nn.Identity()
        
        self.model = model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        img_pil = self._ensure_pil(image)
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        features = self.model(input_tensor).squeeze().cpu().numpy()
        
        if self.use_l2_norm:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
                
        return features

class EfficientNetExtractor(FeatureExtractor):
    """
    Extracts features using a pre-trained EfficientNet-B0 model.
    Removes the final classification layer to return the feature vector (1280-d for B0).
    """
    def __init__(self, use_l2_norm: bool = True, device: str = None):
        self.use_l2_norm = use_l2_norm
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load pre-trained model and remove fully connected layer
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # EfficientNet classifier is a Sequential block, we replace it with Identity
        model.classifier = torch.nn.Identity()
        
        self.model = model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        img_pil = self._ensure_pil(image)
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        features = self.model(input_tensor).squeeze().cpu().numpy()
        
        if self.use_l2_norm:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
                
        return features
