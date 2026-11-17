import time
import os
import json
import numpy as np
import cv2
from typing import Dict, List, Any
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

from core.extractors.base import FeatureExtractor
from core.reducers.base import DimensionalityReducer
from core.data.preprocessor import BoundingBoxCrop
from core.metrics.loss_diagnostics import InformationLossDiagnostics

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
            box = BoundingBoxCrop.parse_box_file(item['box_path'], img.shape[:2])
            img = BoundingBoxCrop.crop_image(img, box)
        vector = self.extractor.extract(img)
        return vector

    def run(self, dataset: List[Dict], test_name: str, output_dir: str = "results/reduction-analysis", run_tag: str = None) -> Dict[str, Any]:
        """
        Executes extraction, reduction and calculates information retention metrics.
        """
        timers = {}
        file_stem = f"{test_name}_{run_tag}_reduction" if run_tag else test_name
        output_json = os.path.join(output_dir, f"{file_stem}.json")
        
        # 1. Feature Extraction
        t0 = time.time()
        vectors = []
        for item in dataset:
            vectors.append(self._process_image(item))
        vectors = np.array(vectors)
        timers['extraction'] = time.time() - t0
        
        # 2. Dimensionality Reduction
        t0 = time.time()
        orig_l2 = getattr(self.reducer, 'use_l2_norm', None)
        if orig_l2 is not None:
            self.reducer.use_l2_norm = False
        try:
            self.reducer.fit(vectors)
            reduced_vectors = self.reducer.transform(vectors)
        finally:
            if orig_l2 is not None:
                self.reducer.use_l2_norm = orig_l2
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
            spearman_r, _ = spearmanr(dist_high, dist_low)
        else:
            r, p_value, r_squared, spearman_r = 0.0, 1.0, 0.0, 0.0
            
        info_retained = r_squared * 100.0
        info_lost = 100.0 - info_retained
        
        # Local neighborhood preservation (Trustworthiness)
        k_trust = min(10, max(1, len(dataset) - 2))
        trustworthiness_score = (
            InformationLossDiagnostics.trustworthiness(vectors, reduced_vectors, n_neighbors=k_trust)
            if len(dataset) > 2 else 1.0
        )
        
        # Check if reducer has explained_variance_ratio_ (e.g. PCA, SVD)
        explained_variance_pct = None
        for attr_obj in (self.reducer, getattr(self.reducer, 'model', None), getattr(self.reducer, 'reducer', None)):
            if attr_obj is not None and hasattr(attr_obj, 'explained_variance_ratio_'):
                explained_variance_pct = float(np.sum(attr_obj.explained_variance_ratio_) * 100.0)
                break
            
        timers['metrics_calculation'] = time.time() - t0
        
        metrics = {
            "test_name": f"{test_name} ({run_tag})" if run_tag else test_name,
            "base_test_name": test_name,
            "run_tag": run_tag,
            "components": {
                "extractor": self.extractor.__class__.__name__,
                "reducer": self.reducer.__class__.__name__
            },
            "dataset_size": len(dataset),
            "information_retained_percent": float(info_retained),
            "information_lost_percent": float(info_lost),
            "pearson_r": float(r),
            "spearman_r": float(spearman_r),
            "trustworthiness_percent": float(trustworthiness_score * 100.0),
            "p_value": float(p_value)
        }
        
        if explained_variance_pct is not None:
            metrics["explained_variance_percent"] = explained_variance_pct
            
        metrics["timers"] = timers
        
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
