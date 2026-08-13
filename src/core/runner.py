import time
import os
import pandas as pd
from typing import Dict, List, Any
import numpy as np
import cv2

from core.extractors.base import FeatureExtractor
from core.reducers.base import DimensionalityReducer
from core.clustering.base import Clusterer
from core.index.searcher import IndexSearcher
from core.data.preprocessor import BoundingBoxCrop

class PipelineRunner:
    """
    Unifies the CBIR execution pipeline.
    """
    def __init__(self, 
                 extractor: FeatureExtractor,
                 reducer: DimensionalityReducer,
                 clusterer: Clusterer,
                 index_searcher: IndexSearcher):
        self.extractor = extractor
        self.reducer = reducer
        self.clusterer = clusterer
        self.index_searcher = index_searcher

    def _process_image(self, item: Dict) -> np.ndarray:
        # Load image
        img = cv2.imread(item['image_path'])
        if img is None:
            raise ValueError(f"Failed to load image at {item['image_path']}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop if bounding box exists
        if item.get('box_path'):
            box = BoundingBoxCrop.parse_box_file(item['box_path'])
            img = BoundingBoxCrop.crop_image(img, box)
            
        # Extract features
        vector = self.extractor.extract(img)
        return vector

    def run(self, gallery: List[Dict], probe: List[Dict], 
            output_csv: str = "results/output.csv", 
            file_exists_action: str = "append",
            top_k: int = 5) -> Dict[str, Any]:
        """
        Executes the full pipeline and writes results to CSV.
        file_exists_action can be 'append' or 'overwrite'.
        """
        timers = {}
        
        # 1. Feature Extraction (Gallery)
        t0 = time.time()
        gallery_vectors = []
        for item in gallery:
            gallery_vectors.append(self._process_image(item))
        gallery_vectors = np.array(gallery_vectors)
        timers['extraction_gallery'] = time.time() - t0
        
        # 2. Dimensionality Reduction
        t0 = time.time()
        self.reducer.fit(gallery_vectors)
        reduced_gallery = self.reducer.transform(gallery_vectors)
        timers['reduction_fit_transform'] = time.time() - t0
        
        # 3. Clustering
        t0 = time.time()
        self.clusterer.fit(reduced_gallery)
        labels = self.clusterer.predict(reduced_gallery)
        centers = self.clusterer.cluster_centers
        timers['clustering'] = time.time() - t0
        
        # 4. Indexing
        t0 = time.time()
        self.index_searcher.build_index(reduced_gallery, gallery, centers, labels)
        timers['indexing'] = time.time() - t0
        
        # 5. Querying (Probe)
        t0 = time.time()
        results = []
        for p_item in probe:
            p_vec = self._process_image(p_item)
            p_vec_red = self.reducer.transform(np.array([p_vec]))[0]
            
            top_matches = self.index_searcher.search(p_vec_red, top_k=top_k)
            
            # Record results
            for rank, (match_meta, dist) in enumerate(top_matches, start=1):
                results.append({
                    "probe_label": p_item['label'],
                    "probe_image": p_item['filename'],
                    "probe_transformation": p_item.get('transformation', ''),
                    "match_rank": rank,
                    "match_label": match_meta['label'],
                    "match_image": match_meta['filename'],
                    "distance": dist,
                    "is_correct": p_item['label'] == match_meta['label']
                })
        timers['querying'] = time.time() - t0
        
        # 6. Save to CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df = pd.DataFrame(results)
        
        if os.path.exists(output_csv):
            if file_exists_action == "append":
                df.to_csv(output_csv, mode='a', header=False, index=False)
            elif file_exists_action == "overwrite":
                df.to_csv(output_csv, mode='w', header=True, index=False)
            else:
                raise ValueError("file_exists_action must be 'append' or 'overwrite'")
        else:
            df.to_csv(output_csv, mode='w', header=True, index=False)
            
        # 7. Compute Confusion Matrix metrics and save JSON
        import json
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        # Pre-count gallery labels
        gallery_label_counts = {}
        for g in gallery:
            lbl = g['label']
            gallery_label_counts[lbl] = gallery_label_counts.get(lbl, 0) + 1
            
        gallery_size = len(gallery)
        
        top1_correct = 0
        topk_correct = 0
        ap_sum = 0.0
        
        for p_item in probe:
            p_label = p_item['label']
            relevant_in_gallery = gallery_label_counts.get(p_label, 0)
            
            p_results = [r for r in results if r['probe_image'] == p_item['filename']]
            p_results.sort(key=lambda x: x['match_rank'])
            
            retrieved_relevant = sum(1 for r in p_results if r['is_correct'])
            retrieved_total = len(p_results)
            
            tp = retrieved_relevant
            fp = retrieved_total - retrieved_relevant
            fn = relevant_in_gallery - retrieved_relevant
            tn = (gallery_size - relevant_in_gallery) - fp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            
            # IR Metrics per query
            if p_results and p_results[0]['is_correct']:
                top1_correct += 1
            if retrieved_relevant > 0:
                topk_correct += 1
                
            hits = 0
            sum_precisions = 0.0
            for i, r in enumerate(p_results, start=1):
                if r['is_correct']:
                    hits += 1
                    sum_precisions += hits / i
                    
            if relevant_in_gallery > 0:
                ap_sum += sum_precisions / relevant_in_gallery
                
        json_path = os.path.splitext(output_csv)[0] + ".json"
        
        # Calculate derived metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
        
        num_probes = len(probe)
        top1_accuracy = top1_correct / num_probes if num_probes > 0 else 0.0
        topk_accuracy = topk_correct / num_probes if num_probes > 0 else 0.0
        mAP = ap_sum / num_probes if num_probes > 0 else 0.0
        
        metrics_data = {
            "csv_file": os.path.basename(output_csv),
            "pipeline_components": {
                "extractor": self.extractor.__class__.__name__,
                "reducer": self.reducer.__class__.__name__,
                "clusterer": self.clusterer.__class__.__name__,
                "index_searcher": self.index_searcher.__class__.__name__
            },
            "confusion_matrix": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn
            },
            "performance_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "top1_accuracy": top1_accuracy,
                f"top{top_k}_accuracy": topk_accuracy,
                "mAP": mAP
            },
            "timers": timers,
            "top_k": top_k
        }
        
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
            
        return {
            "timers": timers,
            "results": results,
            "metrics": metrics_data
        }
