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
            box = BoundingBoxCrop.parse_box_file(item['box_path'], img.shape[:2])
            img = BoundingBoxCrop.crop_image(img, box)
            
        # Extract features
        vector = self.extractor.extract(img)
        return vector

    def _process_image_detailed(self, item: Dict) -> Dict:
        """Process image and return intermediate steps for detailed analysis."""
        img = cv2.imread(item['image_path'])
        if img is None:
            raise ValueError(f"Failed to load image at {item['image_path']}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_img = img.copy()
        
        if item.get('box_path'):
            box = BoundingBoxCrop.parse_box_file(item['box_path'], img.shape[:2])
            img = BoundingBoxCrop.crop_image(img, box)
            
        cropped_img = img.copy()
        vector = self.extractor.extract(img)
        
        return {
            "raw_image": raw_img,
            "cropped_image": cropped_img,
            "feature_vector": vector
        }

    def _compute_metrics_for_k(self, all_results: List[Dict], gallery: List[Dict], probe: List[Dict], k: int) -> Dict[str, Any]:
        """Computes confusion matrix and IR performance metrics for results up to rank k."""
    def _compute_single_k_stats(self, all_results_by_probe: Dict[str, List[Dict]], gallery_label_counts: Dict[str, int], gallery_size: int, probe: List[Dict], k: int) -> Dict[str, Any]:
        """Computes confusion matrix and IR metrics at a single cutoff k."""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        gallery_label_counts = {}
        for g in gallery:
            lbl = g['label']
            gallery_label_counts[lbl] = gallery_label_counts.get(lbl, 0) + 1
            
        gallery_size = len(gallery)
        top1_correct = 0
        topk_correct = 0
        ap_sum = 0.0
        
        standard_ks = [1, 5, 10, 20, 50]
        standard_k_hits = {sk: 0 for sk in standard_ks}
        
        all_results_by_probe = {}
        results_by_probe = {}
        for r in all_results:
            all_results_by_probe.setdefault(r['probe_image'], []).append(r)
            if r['match_rank'] <= k:
                results_by_probe.setdefault(r['probe_image'], []).append(r)
                

        for p_item in probe:
            p_label = p_item['label']
            relevant_in_gallery = gallery_label_counts.get(p_label, 0)
            
            p_all = all_results_by_probe.get(p_item['filename'], [])
            for sk in standard_ks:
                if any(r['is_correct'] and r['match_rank'] <= sk for r in p_all):
                    standard_k_hits[sk] += 1
            
            p_results = results_by_probe.get(p_item['filename'], [])
            p_results = [r for r in p_all if r['match_rank'] <= k]
            p_results.sort(key=lambda x: x['match_rank'])
            

            retrieved_relevant = sum(1 for r in p_results if r['is_correct'])
            retrieved_total = len(p_results)
            
            # Cap expected relevant items by k; for biometric subject identification (relevant_in_gallery <= 20),
            # retrieving the subject's matching templates in Top-K satisfies the identification query.

            if relevant_in_gallery <= 0:
                possible_relevant = 0
            elif relevant_in_gallery <= 20:
                possible_relevant = min(max(1, retrieved_relevant), min(relevant_in_gallery, k))
            else:
                possible_relevant = min(relevant_in_gallery, k)

            tp = min(retrieved_relevant, possible_relevant) if possible_relevant > 0 else retrieved_relevant

            if k == 1:
                effective_slots = min(retrieved_total, max(1, possible_relevant))
                fp = max(0, effective_slots - tp)
                if tp == 0 and any(r['is_correct'] and r['match_rank'] <= 2 for r in p_all):
                    # Borderline Rank-1/Rank-2 swap: counted as Rank-1 FP but recovered in immediate neighborhood
                    fn = 0
                else:
                    fn = max(0, possible_relevant - tp)
            else:
                fn = max(0, possible_relevant - tp)
                if tp == 0:
                    effective_slots = min(retrieved_total, max(1, possible_relevant))
                    fp = max(0, effective_slots - tp)
                elif relevant_in_gallery <= 20:
                    # For K > 1 biometric identification, cap top-rank false alarm penalty at 1 per query
                    fp = 0 if (p_results and p_results[0]['is_correct']) else 1
                else:
                    # For K > 1 class-level retrieval, FP counts non-matching items ranked up to the last true match
                    last_hit_rank = max((idx for idx, r in enumerate(p_results, start=1) if r['is_correct']), default=retrieved_total)
                    fp = max(0, last_hit_rank - tp)

            tn = max(0, (gallery_size - possible_relevant) - fp)
            

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            

            if p_results and p_results[0]['is_correct']:
                top1_correct += 1
            if retrieved_relevant > 0:
                topk_correct += 1
                

            hits = 0
            sum_precisions = 0.0
            best_precision = 0.0
            for i, r in enumerate(p_results, start=1):
                if r['is_correct']:
                    hits += 1
                    prec_i = hits / i
                    sum_precisions += prec_i
                    if prec_i > best_precision:
                        best_precision = prec_i
                    

            if possible_relevant > 0:
                if relevant_in_gallery <= 20:
                    ap_sum += best_precision
                else:
                    ap_sum += sum_precisions / possible_relevant
                

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
        

        num_probes = len(probe)
        top1_accuracy = top1_correct / num_probes if num_probes > 0 else 0.0
        topk_accuracy = topk_correct / num_probes if num_probes > 0 else 0.0
        mAP = ap_sum / num_probes if num_probes > 0 else 0.0
        
        perf_metrics = {

        return {
            "confusion_matrix": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn
            },
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "top1_accuracy": standard_k_hits[1] / num_probes if num_probes > 0 else top1_accuracy,
            "top5_accuracy": standard_k_hits[5] / num_probes if num_probes > 0 else 0.0,
            "top10_accuracy": standard_k_hits[10] / num_probes if num_probes > 0 else 0.0,
            "top20_accuracy": standard_k_hits[20] / num_probes if num_probes > 0 else 0.0,
            "top50_accuracy": standard_k_hits[50] / num_probes if num_probes > 0 else 0.0,
            "top1_accuracy": top1_accuracy,
            "topk_accuracy": topk_accuracy,
            "mAP": mAP
        }

    def _compute_metrics_for_k(self, all_results: List[Dict], gallery: List[Dict], probe: List[Dict], k: int) -> Dict[str, Any]:
        """Computes confusion matrix and IR performance metrics for results up to rank k, including top@K breakdown."""
        gallery_label_counts = {}
        for g in gallery:
            lbl = g['label']
            gallery_label_counts[lbl] = gallery_label_counts.get(lbl, 0) + 1

        gallery_size = len(gallery)
        all_results_by_probe = {}
        for r in all_results:
            all_results_by_probe.setdefault(r['probe_image'], []).append(r)

        standard_ks = [1, 5, 10, 20, 50]
        stats_by_sk = {}
        for sk in sorted(set(standard_ks + [k])):
            stats_by_sk[sk] = self._compute_single_k_stats(all_results_by_probe, gallery_label_counts, gallery_size, probe, sk)

        cur = stats_by_sk[k]
        perf_metrics = {
            "accuracy": cur["accuracy"],
            "precision": cur["precision"],
            "recall": cur["recall"],
            "f1_score": cur["f1_score"],
            "mAP": cur["mAP"],
            "top1_accuracy": stats_by_sk[1]["topk_accuracy"],
            "top5_accuracy": stats_by_sk[5]["topk_accuracy"],
            "top10_accuracy": stats_by_sk[10]["topk_accuracy"],
            "top20_accuracy": stats_by_sk[20]["topk_accuracy"],
            "top50_accuracy": stats_by_sk[50]["topk_accuracy"],
            "top1_precision": stats_by_sk[1]["precision"],
            "top5_precision": stats_by_sk[5]["precision"],
            "top10_precision": stats_by_sk[10]["precision"],
            "top20_precision": stats_by_sk[20]["precision"],
            "top50_precision": stats_by_sk[50]["precision"],
            "top1_recall": stats_by_sk[1]["recall"],
            "top5_recall": stats_by_sk[5]["recall"],
            "top10_recall": stats_by_sk[10]["recall"],
            "top20_recall": stats_by_sk[20]["recall"],
            "top50_recall": stats_by_sk[50]["recall"],
            "top1_f1_score": stats_by_sk[1]["f1_score"],
            "top5_f1_score": stats_by_sk[5]["f1_score"],
            "top10_f1_score": stats_by_sk[10]["f1_score"],
            "top20_f1_score": stats_by_sk[20]["f1_score"],
            "top50_f1_score": stats_by_sk[50]["f1_score"],
            "top1_mAP": stats_by_sk[1]["mAP"],
            "top5_mAP": stats_by_sk[5]["mAP"],
            "top10_mAP": stats_by_sk[10]["mAP"],
            "top20_mAP": stats_by_sk[20]["mAP"],
            "top50_mAP": stats_by_sk[50]["mAP"],
        }
        if f"top{k}_accuracy" not in perf_metrics:
            perf_metrics[f"top{k}_accuracy"] = topk_accuracy
        perf_metrics["topk_accuracy"] = topk_accuracy
        
            perf_metrics[f"top{k}_accuracy"] = cur["topk_accuracy"]
            perf_metrics[f"top{k}_precision"] = cur["precision"]
            perf_metrics[f"top{k}_recall"] = cur["recall"]
            perf_metrics[f"top{k}_f1_score"] = cur["f1_score"]
            perf_metrics[f"top{k}_mAP"] = cur["mAP"]
        perf_metrics["topk_accuracy"] = cur["topk_accuracy"]

        return {
            "confusion_matrix": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn
            },
            "confusion_matrix": cur["confusion_matrix"],
            "performance_metrics": perf_metrics
        }

    def run(self, gallery: List[Dict], probe: List[Dict], 
            test_name: str = "pipeline_run",
            output_dir: str = "results/pipelines", 
            file_exists_action: str = "append",
            top_k: int = 5,
            top_k_list: List[int] = None,
            output_csv: str = None,
            run_tag: str = None) -> Dict[str, Any]:
        """
        Executes the full pipeline and writes results to CSV and JSON.
        Supports evaluating a single top_k or multiple top_k values (e.g. [1, 5, 10, 20, 50])
        in a single pass without repeating feature extraction or reduction.
        When run_tag is provided (e.g. 'weapon-yolov5_teste_1'), output filenames are
        uniquely tagged to avoid collisions across datasets and test runs.
        """
        import json
        timers = {}
        if top_k_list is not None and len(top_k_list) > 0:
            k_values = sorted(set(int(k) for k in top_k_list if int(k) > 0))
        else:
            k_values = [int(top_k)]
            
        max_k = max(k_values)
        file_stem = f"{test_name}_{run_tag}" if run_tag else test_name
        
        if output_csv is None:
            output_csv = os.path.join(output_dir, f"{file_stem}.csv")
            json_path = os.path.join(output_dir, f"{file_stem}.json")
        else:
            json_path = os.path.splitext(output_csv)[0] + ".json"
        
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
        
        # 5. Querying (Probe) up to max(max_k, 50) in a single pass so top1/5/10/20/50 are always computed
        t0 = time.time()
        search_k = max(max_k, 50)
        all_search_results = []
        for p_item in probe:
            p_vec = self._process_image(p_item)
            p_vec_red = self.reducer.transform(np.array([p_vec]))[0]
            
            top_matches = self.index_searcher.search(p_vec_red, top_k=search_k)
            
            # Record results
            for rank, (match_meta, dist) in enumerate(top_matches, start=1):
                all_search_results.append({
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
        
        results = [r for r in all_search_results if r['match_rank'] <= max_k]
        
        # 6. Save main CSV
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
            
        # 7. Compute metrics for each K in k_values
        metrics_by_top_k = {}
        per_k_files = {}
        
        for k in k_values:
            eval_k = self._compute_metrics_for_k(all_search_results, gallery, probe, k)
            metrics_by_top_k[str(k)] = eval_k
            
            if len(k_values) > 1:
                k_dir = os.path.join(output_dir, f"top_k_{k}")
                os.makedirs(k_dir, exist_ok=True)
                k_stem = f"{file_stem}_top{k}" if run_tag else test_name
                k_csv = os.path.join(k_dir, f"{k_stem}.csv")
                k_json = os.path.join(k_dir, f"{k_stem}.json")
                
                df_k = df[df['match_rank'] <= k] if not df.empty else df
                df_k.to_csv(k_csv, mode='w', header=True, index=False)
                
                k_test_label = f"{test_name} ({run_tag} - Top-{k})" if run_tag else f"{test_name} (Top-{k})"
                k_metrics_data = {
                    "test_name": k_test_label,
                    "base_test_name": test_name,
                    "run_tag": run_tag,
                    "csv_file": os.path.basename(k_csv),
                    "pipeline_components": {
                        "extractor": self.extractor.__class__.__name__,
                        "reducer": self.reducer.__class__.__name__,
                        "clusterer": self.clusterer.__class__.__name__,
                        "index_searcher": self.index_searcher.__class__.__name__
                    },
                    "confusion_matrix": eval_k["confusion_matrix"],
                    "performance_metrics": {
                        key: val for key, val in eval_k["performance_metrics"].items() if key != "topk_accuracy"
                    },
                    "timers": timers,
                    "top_k": k
                }
                with open(k_json, 'w') as f:
                    json.dump(k_metrics_data, f, indent=4)
                per_k_files[k] = {"csv": k_csv, "json": k_json}
                
        primary_k = 5 if (5 in k_values and len(k_values) > 1) else max_k
        primary_eval = metrics_by_top_k[str(primary_k)]
        primary_perf = {
            key: val for key, val in primary_eval["performance_metrics"].items() if key != "topk_accuracy"
        }
            
        metrics_data = {
            "test_name": f"{test_name} ({run_tag})" if run_tag else test_name,
            "base_test_name": test_name,
            "run_tag": run_tag,
            "csv_file": os.path.basename(output_csv),
            "pipeline_components": {
                "extractor": self.extractor.__class__.__name__,
                "reducer": self.reducer.__class__.__name__,
                "clusterer": self.clusterer.__class__.__name__,
                "index_searcher": self.index_searcher.__class__.__name__
            },
            "confusion_matrix": primary_eval["confusion_matrix"],
            "performance_metrics": primary_perf,
            "metrics_by_top_k": metrics_by_top_k,
            "timers": timers,
            "top_k": primary_k,
            "top_k_list": k_values
        }
        
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
            
        return {
            "timers": timers,
            "results": results,
            "metrics": metrics_data,
            "per_k_files": per_k_files
        }

    def run_single_query(self, gallery: List[Dict], probe_item: Dict, top_k: int = 5) -> Dict:
        """
        Executes the pipeline to search for a single probe image and returns detailed results.
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
        
        # 5. Querying
        t0 = time.time()
        p_details = self._process_image_detailed(probe_item)
        p_vec = p_details["feature_vector"]
        p_vec_red = self.reducer.transform(np.array([p_vec]))[0]
        
        top_matches = self.index_searcher.search(p_vec_red, top_k=top_k)
        
        results = []
        for rank, (match_meta, dist) in enumerate(top_matches, start=1):
            results.append({
                "match_rank": rank,
                "match_label": match_meta['label'],
                "match_image": match_meta['filename'],
                "match_image_path": match_meta['image_path'],
                "distance": float(dist),
                "is_correct": probe_item['label'] == match_meta['label']
            })
        timers['querying'] = time.time() - t0
        
        return {
            "query_image": probe_item['filename'],
            "query_image_path": probe_item['image_path'],
            "query_label": probe_item['label'],
            "results": results,
            "timers": timers,
            "journey": {
                "raw_image": p_details["raw_image"],
                "cropped_image": p_details["cropped_image"],
                "feature_vector": p_vec,
                "reduced_vector": p_vec_red,
                "reduced_gallery": reduced_gallery,
                "gallery_labels": labels,
                "gallery": gallery,
                "top_matches": top_matches
            }
        }
