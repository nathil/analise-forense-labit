import time
import os
import json
import numpy as np
import cv2
from typing import Dict, List, Any
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from core.extractors.base import FeatureExtractor
from core.reducers.base import DimensionalityReducer
from core.data.preprocessor import BoundingBoxCrop

class ReductionAnalyzerRunner:
    """
    Analyzes dimensionality reduction techniques by quantifying 
    the percentage of information retained and lost.
    """
    def __init__(self, extractor: FeatureExtractor, reducer: DimensionalityReducer):
        self.extractor = extractor
        self.reducer = reducer
        
    def _process_image(self, item: Dict) -> np.ndarray:
        img = cv2.imread(item['image_path'])
        if img is None:
            raise ValueError(f"Failed to load image at {item['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if item.get('box_path'):
            box = BoundingBoxCrop.parse_box_file(item['box_path'])
            img = BoundingBoxCrop.crop_image(img, box)
        vector = self.extractor.extract(img)
        return vector

    def run(self, dataset: List[Dict], output_json: str) -> Dict[str, Any]:
        """
        Executes extraction, reduction and calculates information retention metrics.
        """
        timers = {}
        
        # 1. Feature Extraction
        t0 = time.time()
        vectors = []
        for item in dataset:
            vectors.append(self._process_image(item))
        vectors = np.array(vectors)
        timers['extraction'] = time.time() - t0
        
        # 2. Dimensionality Reduction
        t0 = time.time()
        self.reducer.fit(vectors)
        reduced_vectors = self.reducer.transform(vectors)
        timers['reduction'] = time.time() - t0
        
        # 3. Calculate Information Metrics
        t0 = time.time()
        # Compute pairwise distances (upper triangle flattened)
        dist_high = pdist(vectors, metric='euclidean')
        dist_low = pdist(reduced_vectors, metric='euclidean')
        
        # Pearson correlation (Mantel test equivalent)
        # Avoid zero variance issues if dataset is too small or constant
        if len(np.unique(dist_high)) > 1 and len(np.unique(dist_low)) > 1:
            r, p_value = pearsonr(dist_high, dist_low)
            r_squared = r ** 2
        else:
            r, p_value, r_squared = 0.0, 1.0, 0.0
            
        info_retained = r_squared * 100.0
        info_lost = 100.0 - info_retained
        
        # Check if reducer has explained_variance_ratio_ (e.g. PCA, SVD)
        explained_variance_pct = None
        if hasattr(self.reducer, 'reducer') and hasattr(self.reducer.reducer, 'explained_variance_ratio_'):
            explained_variance_pct = float(np.sum(self.reducer.reducer.explained_variance_ratio_) * 100.0)
            
        timers['metrics_calculation'] = time.time() - t0
        
        metrics = {
            "components": {
                "extractor": self.extractor.__class__.__name__,
                "reducer": self.reducer.__class__.__name__
            },
            "dataset_size": len(dataset),
            "information_retained_percent": float(info_retained),
            "information_lost_percent": float(info_lost),
            "pearson_r": float(r),
            "p_value": float(p_value)
        }
        
        if explained_variance_pct is not None:
            metrics["explained_variance_percent"] = explained_variance_pct
            
        metrics["timers"] = timers
        
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
