import os
import time
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable

from core.extractors.base import FeatureExtractor
from core.data.preprocessor import BoundingBoxCrop


class GridSearchRunner:
    """
    Executes a combinatorial Grid Search across all combinations of
    Feature Extractors, Dimensionality Reducers, Clusterers, and Index Searchers.
    
    Optimized with hierarchical tiered caching to minimize redundant computations.
    """
    def __init__(self,
                 extractors: Dict[str, Any],
                 reducers: Dict[str, Any],
                 clusterers: Dict[str, Any],
                 searchers: Dict[str, Callable],
                 device: str = "cpu",
                 top_k: int = 5):
        self.extractors = extractors
        self.reducers = reducers
        self.clusterers = clusterers
        self.searchers = searchers
        self.device = device
        self.top_k = top_k

    def _process_image(self, extractor: FeatureExtractor, item: Dict) -> np.ndarray:
        img = cv2.imread(item['image_path'])
        if img is None:
            raise ValueError(f"Failed to load image at {item['image_path']}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if item.get('box_path'):
            box = BoundingBoxCrop.parse_box_file(item['box_path'])
            img = BoundingBoxCrop.crop_image(img, box)
            
        return extractor.extract(img)

    def run(self, gallery: List[Dict], probe: List[Dict], 
            test_name: str,
            output_dir: str = "results/grid-search") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs all combinatorial pipelines and writes the consolidated query results to CSV.
        Returns (summary_metrics_df, best_pipeline_info).
        """
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"{test_name}.csv")

        total_combinations = len(self.extractors) * len(self.reducers) * len(self.clusterers) * len(self.searchers)
        print(f"\n[+] Starting Grid Search over {total_combinations} combinations...")
        print(f"    - Extractors ({len(self.extractors)}): {list(self.extractors.keys())}")
        print(f"    - Reducers ({len(self.reducers)}): {list(self.reducers.keys())}")
        print(f"    - Clusterers ({len(self.clusterers)}): {list(self.clusterers.keys())}")
        print(f"    - Searchers ({len(self.searchers)}): {list(self.searchers.keys())}")
        
        gallery_size = len(gallery)
        num_probes = len(probe)
        
        gallery_label_counts = {}
        for g in gallery:
            lbl = g['label']
            gallery_label_counts[lbl] = gallery_label_counts.get(lbl, 0) + 1

        all_query_rows = []
        summary_rows = []
        comb_idx = 0
        grid_start_time = time.time()

        for ext_name, ext_cls in self.extractors.items():
            print(f"\n[+] [Tier 1] Extracting features with {ext_name}...")
            t0 = time.time()
            if ext_name in ["ResNet50", "EfficientNet"]:
                extractor = ext_cls(use_l2_norm=False, device=self.device)
            else:
                extractor = ext_cls(device=self.device)

            gallery_vectors = np.array([self._process_image(extractor, item) for item in gallery])
            probe_vectors = np.array([self._process_image(extractor, item) for item in probe])
            ext_gallery_time = time.time() - t0
            
            # Clean up extractor instance to save GPU/CPU memory
            del extractor

            for red_name, red_cls in self.reducers.items():
                t0 = time.time()
                red_k = min(128, max(1, len(gallery) - 2))
                if red_name in ["PCA", "SVD"]:
                    reducer = red_cls(n_components=red_k)
                elif red_name == "UMAP":
                    reducer = red_cls(n_components=red_k, metric='euclidean', n_neighbors=min(15, max(2, len(gallery)-1)))
                else:
                    reducer = red_cls()

                reducer.fit(gallery_vectors)
                reduced_gallery = reducer.transform(gallery_vectors)
                reduced_probe = reducer.transform(probe_vectors)
                red_time = time.time() - t0

                for clust_name, clust_cls in self.clusterers.items():
                    t0 = time.time()
                    if clust_name == "MiniBatchKMeans":
                        k_clust = min(8, max(2, len(gallery)//2))
                        clusterer = clust_cls(n_clusters=k_clust, metric='euclidean')
                    elif clust_name == "HDBSCAN":
                        clusterer = clust_cls(min_cluster_size=max(2, min(5, len(gallery)//2)), metric='euclidean')
                    elif clust_name == "Louvain":
                        clusterer = clust_cls(n_neighbors=min(5, max(2, len(gallery)-1)), metric='cosine')
                    else:
                        clusterer = clust_cls()

                    clusterer.fit(reduced_gallery)
                    labels = clusterer.predict(reduced_gallery)
                    centers = clusterer.cluster_centers
                    clust_time = time.time() - t0

                    for search_name, search_factory in self.searchers.items():
                        comb_idx += 1
                        s_short = search_name.split(" ")[0] if "(" in search_name else search_name
                        s_metric = "Cosine" if "Cosine" in search_name else "Euclidean"
                        pipe_name = f"{ext_name}_{red_name}_{clust_name}_{s_metric}"

                        t0 = time.time()
                        searcher = search_factory()
                        searcher.build_index(reduced_gallery, gallery, centers, labels)
                        idx_time = time.time() - t0

                        t0 = time.time()
                        pipe_results = []
                        for p_idx, p_item in enumerate(probe):
                            p_vec_red = reduced_probe[p_idx]
                            top_matches = searcher.search(p_vec_red, top_k=self.top_k)

                            for rank, (match_meta, dist) in enumerate(top_matches, start=1):
                                is_correct = (p_item['label'] == match_meta['label'])
                                row = {
                                    "pipeline_name": pipe_name,
                                    "extractor": ext_name,
                                    "reducer": red_name,
                                    "clusterer": clust_name,
                                    "searcher": s_metric,
                                    "probe_label": p_item['label'],
                                    "probe_image": p_item['filename'],
                                    "probe_transformation": p_item.get('transformation', ''),
                                    "match_rank": rank,
                                    "match_label": match_meta['label'],
                                    "match_image": match_meta['filename'],
                                    "distance": float(dist),
                                    "is_correct": is_correct
                                }
                                all_query_rows.append(row)
                                pipe_results.append(row)
                                
                        query_time = time.time() - t0

                        # Calculate IR and Classification Metrics
                        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
                        top1_correct = 0
                        topk_correct = 0
                        ap_sum = 0.0

                        for p_item in probe:
                            p_label = p_item['label']
                            relevant_in_gallery = gallery_label_counts.get(p_label, 0)
                            p_res = [r for r in pipe_results if r['probe_image'] == p_item['filename']]
                            p_res.sort(key=lambda x: x['match_rank'])

                            retrieved_rel = sum(1 for r in p_res if r['is_correct'])
                            retrieved_total = len(p_res)

                            tp = retrieved_rel
                            fp = retrieved_total - retrieved_rel
                            fn = relevant_in_gallery - retrieved_rel
                            tn = (gallery_size - relevant_in_gallery) - fp

                            total_tp += tp
                            total_fp += fp
                            total_fn += fn
                            total_tn += tn

                            if p_res and p_res[0]['is_correct']:
                                top1_correct += 1
                            if retrieved_rel > 0:
                                topk_correct += 1

                            hits = 0
                            sum_p = 0.0
                            for i, r in enumerate(p_res, start=1):
                                if r['is_correct']:
                                    hits += 1
                                    sum_p += hits / i
                            if relevant_in_gallery > 0:
                                ap_sum += sum_p / relevant_in_gallery

                        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
                        top1_acc = top1_correct / num_probes if num_probes > 0 else 0.0
                        topk_acc = topk_correct / num_probes if num_probes > 0 else 0.0
                        mAP = ap_sum / num_probes if num_probes > 0 else 0.0
                        
                        total_pipe_time = ext_gallery_time + red_time + clust_time + idx_time + query_time

                        summary_rows.append({
                            "pipeline_name": pipe_name,
                            "extractor": ext_name,
                            "reducer": red_name,
                            "clusterer": clust_name,
                            "searcher": s_metric,
                            "mAP": mAP,
                            "top1_accuracy": top1_acc,
                            f"top{self.top_k}_accuracy": topk_acc,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1_score,
                            "accuracy": accuracy,
                            "time_extraction": ext_gallery_time,
                            "time_reduction": red_time,
                            "time_clustering": clust_time,
                            "time_indexing": idx_time,
                            "time_querying": query_time,
                            "total_time": total_pipe_time
                        })

                        print(f"    [{comb_idx}/{total_combinations}] Evaluated: {pipe_name} (mAP: {mAP:.4f}, Top-1: {top1_acc*100:.1f}%)")

        total_grid_duration = time.time() - grid_start_time
        print(f"\n[+] Grid Search completed in {total_grid_duration:.2f}s!")

        # 1. Save Massive Single CSV
        print(f"[+] Writing consolidated query CSV ({len(all_query_rows)} rows) to {output_csv}...")
        df_all = pd.DataFrame(all_query_rows)
        df_all.to_csv(output_csv, index=False)

        # 2. Build Summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        summary_df.sort_values(by="mAP", ascending=False, inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

        best_pipeline = summary_df.iloc[0].to_dict()

        return summary_df, best_pipeline
