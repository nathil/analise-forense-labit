import os
import uuid
import sys
import cv2
import numpy as np
from datetime import datetime

try:
    import questionary
except ImportError:
    print("Please install questionary using 'uv add questionary'")
    sys.exit(1)

from core.data.loader import BivTattDatasetLoader, DatasetSplitter
from core.runner import PipelineRunner
from core.reduction_analyzer import ReductionAnalyzerRunner

from core.extractors.cnn import ResNet50Extractor, EfficientNetExtractor
from core.extractors.vit import DINOv2Extractor, ViTExtractor
from core.extractors.clip import CLIPExtractor

from core.reducers.pca import PCAReducer, SVDReducer
from core.reducers.umap_proj import UMAPReducer
from core.reducers.passthrough import PassThroughReducer
from core.reducers.graph import KNNGraphReducer

from core.clustering.kmeans import MiniBatchKMeansClusterer
from core.clustering.hdbscan import HDBSCANClusterer
from core.clustering.graph import LouvainClusterer

from core.index.searcher import IndexSearcher

EXTRACTORS = {
    "ResNet50": ResNet50Extractor,
    "EfficientNet": EfficientNetExtractor,
    "ViT": ViTExtractor,
    "DINOv2": DINOv2Extractor,
    "CLIP": CLIPExtractor
}

REDUCERS = {
    "PCA": PCAReducer,
    "SVD": SVDReducer,
    "UMAP": UMAPReducer,
    "PassThrough": PassThroughReducer,
    "KNN Graph": KNNGraphReducer
}

CLUSTERERS = {
    "MiniBatchKMeans": MiniBatchKMeansClusterer,
    "HDBSCAN": HDBSCANClusterer,
    "Louvain": LouvainClusterer
}

SEARCHERS = {
    "IndexSearcher (Euclidean)": lambda: IndexSearcher(metric='euclidean'),
    "IndexSearcher (Cosine)": lambda: IndexSearcher(metric='cosine'),
    "IndexSearcher (Dot Product)": lambda: IndexSearcher(metric='dot')
}


def create_dummy_dataset(output_dir="data/dummy_dataset", n_gallery=20, n_probe=5):
    """Creates a dummy BivTatt dataset to test the pipelines."""
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    gallery = []
    probe = []
    
    for i in range(n_gallery):
        img_path = f"{output_dir}/images/lbl{i}_{i}.jpg"
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        gallery.append({
            "label": f"lbl{i}", "filename": f"lbl{i}_{i}.jpg", 
            "image_path": img_path, "box_path": None, "is_original": True
        })
        
    for i in range(n_probe):
        img_path = f"{output_dir}/images/lbl{i}_{i}_trans.jpg"
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        probe.append({
            "label": f"lbl{i}", "filename": f"lbl{i}_{i}_trans.jpg", 
            "image_path": img_path, "box_path": None, "is_original": False,
            "transformation": "trans"
        })
        
    return gallery, probe


def is_valid_int(val):
    if not val.strip():
        return True
    try:
        if int(val) < 0:
            return "Please enter a positive integer."
        return True
    except ValueError:
        return "Please enter a valid integer or leave empty."


def run_tui():
    print("=" * 60)
    print(" ForensTatt - Pipeline Runner")
    print("=" * 60)
    
    mode = questionary.select(
        "Select the mode of operation:",
        choices=[
            "1. Full Pipeline Execution",
            "2. Single Query Execution",
            "3. Grid Search (All Combinations)",
            "4. Dimensionality Reduction Analysis",
            "5. Clear Results & Data"
        ]
    ).ask()
    
    if not mode:
        return
        
    if "5." in mode:
        from core.cleaner import WorkspaceCleaner
        targets = questionary.checkbox(
            "Select which folders to clear:",
            choices=[
                "data",
                "pipelines",
                "reduction-analysis",
                "charts",
                "multiple-pipelines",
                "pipeline",
                "query",
                "grid-search"
            ]
        ).ask()
        
        if not targets:
            return
            
        confirm = questionary.confirm(
            f"WARNING: This will permanently delete files in: {', '.join(targets)}. Are you sure?",
            default=False
        ).ask()
        
        if confirm:
            WorkspaceCleaner.clean(targets)
        else:
            print("Operation cancelled.")
        return
        
    is_grid_search = "3." in mode
    is_reduction_analysis = "4." in mode
    is_single_query = "2." in mode

    # 1. Dataset Selection
    dataset_source = questionary.select(
        "Select the dataset source:",
        choices=[
            "Kagglehub (mexwell/bivtatt-dataset)",
            "Dummy Dataset (Generated for fast testing)",
            "Local Directory"
        ]
    ).ask()
    
    if dataset_source is None:
        return

    dataset_path = None
    gallery = []
    probe = []
    dataset_full = []
    
    # 2. Resolve and Load Dataset Full Metadata
    if "Dummy" in dataset_source:
        print("Creating dummy dataset...")
        g, p = create_dummy_dataset()
        dataset_full = g + p
    else:
        if "Local" in dataset_source:
            dataset_path = questionary.path("Enter path to the local BivTatt directory:").ask()
            if not dataset_path: return
            
        elif "Kagglehub" in dataset_source:
            print("Resolving dataset via Kagglehub...")
            import kagglehub
            os.environ['KAGGLEHUB_CACHE'] = os.path.join(os.getcwd(), 'data', 'kagglehub')
            dataset_path = kagglehub.dataset_download("mexwell/bivtatt-dataset")
            if not os.path.exists(os.path.join(dataset_path, "images")):
                subdirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d))]
                for subdir in subdirs:
                    if os.path.exists(os.path.join(subdir, "images")):
                        dataset_path = subdir
                        break
        
        if dataset_path:
            loader = BivTattDatasetLoader(dataset_path)
            dataset_full = loader.load_dataset(start=0, length=None)
        else:
            return

    total_items = len(dataset_full)
    print(f"\n[+] Dataset resolved. Total items available: {total_items}")

    if total_items == 0:
        print("Dataset is empty. Exiting.")
        return

    # 3. Partitioning
    start_str = questionary.text(f"Dataset partition start index (0 to {total_items - 1}, default 0):", validate=is_valid_int).ask()
    if start_str is None: return
    start_idx = int(start_str) if start_str.strip() else 0
    
    max_length = total_items - start_idx
    length_str = questionary.text(f"Dataset partition length (max {max_length}, default all):", validate=is_valid_int).ask()
    if length_str is None: return
    length_idx = int(length_str) if length_str.strip() else None

    # Slice and Split
    if length_idx is not None:
        dataset = dataset_full[start_idx:start_idx+length_idx]
    else:
        dataset = dataset_full[start_idx:]
        
    splitter = DatasetSplitter()
    gallery, probe = splitter.split(dataset)
    print(f"Loaded {len(gallery)} gallery images and {len(probe)} probe images for this run.")

    if not gallery or not probe:
        print("Partition does not have enough gallery/probe splits. Exiting.")
        return

    # 2.5 Single Query Parameters
    probe_index = None
    top_k = 5
    if is_single_query:
        if len(probe) == 0:
            print("Error: No probe images loaded. Cannot run Single Query.")
            return
        while True:
            idx_str = questionary.text(f"Select a probe image index to use as the single query (0 to {len(probe)-1}):", default="0").ask()
            if not idx_str: return
            try:
                probe_index = int(idx_str)
                if 0 <= probe_index < len(probe):
                    break
                else:
                    print(f"Index out of range. Please enter a value between 0 and {len(probe)-1}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
                
        while True:
            k_str = questionary.text("Enter Top-K results to retrieve (default 5):", default="5").ask()
            if not k_str: return
            try:
                top_k = int(k_str)
                if top_k > 0: break
            except ValueError:
                print("Invalid input.")

    # 3. Component Selection
    device = questionary.select("Select computation device:", choices=["cuda", "cpu"]).ask()
    if not device: return
    
    if is_grid_search:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{timestamp}_grid_search"
        test_name_input = questionary.text(f"Enter a name for this Grid Search run (leave blank to use: {default_name}):").ask()
        if test_name_input is None: return
        test_name = test_name_input.strip() or default_name

        confirm = questionary.confirm(f"Execute Grid Search over all {len(EXTRACTORS)*len(REDUCERS)*len(CLUSTERERS)*len(SEARCHERS)} pipeline combinations?").ask()
        if not confirm: return

        from core.grid_search import GridSearchRunner
        from core.visualization.chart_generator import ChartGenerator

        chart_gen = ChartGenerator(output_dir="results/charts")
        grid_runner = GridSearchRunner(
            extractors=EXTRACTORS,
            reducers=REDUCERS,
            clusterers=CLUSTERERS,
            searchers=SEARCHERS,
            device=device,
            top_k=5
        )

        summary_df, best_pipe = grid_runner.run(gallery, probe, test_name=test_name)
        chart_gen.generate_grid_search_charts(test_name, summary_df)

        print("\n" + "=" * 60)
        print(f" 🏆 GRID SEARCH WINNER: {best_pipe['pipeline_name']}")
        print(f"    - Mean Average Precision (mAP): {best_pipe['mAP']:.4f}")
        print(f"    - Top-1 Accuracy:               {best_pipe['top1_accuracy']*100:.1f}%")
        print(f"    - Top-5 Accuracy:               {best_pipe['top5_accuracy']*100:.1f}%")
        print(f"    - Precision:                    {best_pipe['precision']:.4f}")
        print(f"    - Recall:                       {best_pipe['recall']:.4f}")
        print(f"    - F1-Score:                     {best_pipe['f1_score']:.4f}")
        print(f"    - Total Pipeline Time:          {best_pipe['total_time']:.3f}s")
        print(f"    - Consolidated CSV:             results/grid-search/{test_name}.csv")
        print(f"    - Analytical Charts:            results/charts/grid-search/")
        print("=" * 60 + "\n")
        return

    pipelines_to_run = []
    
    config_mode = questionary.select(
        "Select Pipeline Configuration Mode:",
        choices=[
            "Load predefined from default_pipelines.json",
            "Manual/Custom Selection"
        ]
    ).ask()
    
    if not config_mode: return
    
    if "Load predefined" in config_mode:
        try:
            import json
            with open("default_pipelines.json", "r") as f:
                predefined = json.load(f)
            
            for p in predefined:
                if is_reduction_analysis:
                    pipelines_to_run.append({
                        "extractor": p["extractor"],
                        "reducer": p["reducer"],
                        "test_name": p.get("test_name", f"{p['extractor']}_{p['reducer']}")
                    })
                else:
                    pipelines_to_run.append({
                        "extractor": p["extractor"],
                        "reducer": p["reducer"],
                        "clusterer": p["clusterer"],
                        "searcher": p["searcher"],
                        "test_name": p.get("test_name", f"{p['extractor']}_{p['reducer']}_{p['clusterer']}_{p['searcher']}")
                    })
            print(f"Loaded {len(pipelines_to_run)} predefined pipelines.")
        except Exception as e:
            print(f"Error loading predefined pipelines: {e}")
            return
    else:
        if is_reduction_analysis:
            extractor_choice = questionary.select("Select Feature Extractor:", choices=list(EXTRACTORS.keys())).ask()
            if not extractor_choice: return
            reducer_choice = questionary.select("Select Dimensionality Reducer:", choices=list(REDUCERS.keys())).ask()
            if not reducer_choice: return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"{timestamp}_{extractor_choice}_{reducer_choice}"
            test_name_input = questionary.text(f"Enter a name for this run (leave blank to use: {default_name}):").ask()
            if test_name_input is None: return
            test_name = test_name_input.strip() or default_name
            
            pipelines_to_run.append({
                "extractor": extractor_choice,
                "reducer": reducer_choice,
                "test_name": test_name
            })
        else:
            while True:
                extractor_choice = questionary.select("Select Feature Extractor:", choices=list(EXTRACTORS.keys())).ask()
                if not extractor_choice: return
                reducer_choice = questionary.select("Select Dimensionality Reducer:", choices=list(REDUCERS.keys())).ask()
                if not reducer_choice: return
                clusterer_choice = questionary.select("Select Clusterer:", choices=list(CLUSTERERS.keys())).ask()
                if not clusterer_choice: return
                searcher_choice = questionary.select("Select Index Searcher:", choices=list(SEARCHERS.keys())).ask()
                if not searcher_choice: return
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s_choice = searcher_choice.split(" ")[0] if searcher_choice else "Searcher"
                default_name = f"{timestamp}_{extractor_choice}_{reducer_choice}_{clusterer_choice}_{s_choice}"
                test_name_input = questionary.text(f"Enter a name for this run (leave blank to use: {default_name}):").ask()
                if test_name_input is None: return
                test_name = test_name_input.strip() or default_name
                
                pipelines_to_run.append({
                    "extractor": extractor_choice,
                    "reducer": reducer_choice,
                    "clusterer": clusterer_choice,
                    "searcher": searcher_choice,
                    "test_name": test_name
                })
                
                add_more = questionary.confirm("Add another pipeline to compare?").ask()
                if not add_more:
                    break
                
    batch_name = None
    if not is_reduction_analysis and len(pipelines_to_run) > 1:
        default_batch = f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_input = questionary.text(f"Enter a name for this comparative batch (leave blank to use: {default_batch}):").ask()
        if batch_input is None: return
        batch_name = batch_input.strip() or default_batch
    elif is_single_query and len(pipelines_to_run) == 1:
        batch_name = pipelines_to_run[0]["test_name"]

    # 5. Confirmation
    confirm = questionary.confirm("Execute?").ask()
    if not confirm:
        return

    print(f"\n[+] Executing {len(pipelines_to_run)} pipeline(s)...")
    
    csv_paths = []
    json_paths = []
    test_names = []
    from core.visualization.chart_generator import ChartGenerator
    chart_gen = ChartGenerator(output_dir="results/charts")
    
    for i, p_config in enumerate(pipelines_to_run):
        t_name = p_config["test_name"]
        print(f"\n[{i+1}/{len(pipelines_to_run)}] Building components for {t_name}...")
        
        if p_config["extractor"] == "ResNet50":
            extractor = EXTRACTORS[p_config["extractor"]](use_l2_norm=False, device=device)
        else:
            extractor = EXTRACTORS[p_config["extractor"]](device=device)
        
        red_k = min(128, max(1, len(gallery) - 2))
        if p_config["reducer"] in ["PCA", "SVD"]:
            reducer = REDUCERS[p_config["reducer"]](n_components=red_k)
        elif p_config["reducer"] == "UMAP":
            reducer = REDUCERS[p_config["reducer"]](n_components=red_k, metric='euclidean', n_neighbors=15)
        else:
            reducer = REDUCERS[p_config["reducer"]]()

        if is_reduction_analysis:
            print(f"\n[+] Running Reduction Analysis ({t_name})...")
            runner = ReductionAnalyzerRunner(extractor, reducer)
            out = runner.run(dataset_full, test_name=t_name)
            
            output_json = os.path.join("results/reduction-analysis", f"{t_name}.json")
            print(f"\n[+] Execution finished! Results saved to {output_json}")
            print("-" * 40)
            print(f"Information Retained: {out['information_retained_percent']:.2f}%")
            print(f"Information Lost:     {out['information_lost_percent']:.2f}%")
            if 'explained_variance_percent' in out:
                print(f"Explained Variance:   {out['explained_variance_percent']:.2f}%")
            print("-" * 40)
            
        else:
            if p_config["clusterer"] == "MiniBatchKMeans":
                n_clusters = max(2, min(8, len(gallery) // 2)) if len(gallery) > 2 else len(gallery)
                clusterer = CLUSTERERS[p_config["clusterer"]](n_clusters=n_clusters, metric='euclidean')
            elif p_config["clusterer"] == "HDBSCAN":
                clusterer = CLUSTERERS[p_config["clusterer"]](min_cluster_size=max(2, min(5, len(gallery)//2)), metric='euclidean')
            elif p_config["clusterer"] == "Louvain":
                clusterer = CLUSTERERS[p_config["clusterer"]](n_neighbors=5, metric='cosine')
    
            searcher = SEARCHERS[p_config["searcher"]]()
    
            runner = PipelineRunner(extractor, reducer, clusterer, searcher)
            if is_single_query:
                print(f"\n[+] Running Single Query pipeline ({t_name})...")
                out = runner.run_single_query(gallery, probe[probe_index], top_k=top_k)
                
                j = out["journey"]
                v_raw = j["feature_vector"]
                v_red = j["reduced_vector"]
                print(f"    -> Raw Feature Vector : Dim {len(v_raw)}, Min: {v_raw.min():.4f}, Max: {v_raw.max():.4f}, Mean: {v_raw.mean():.4f}")
                print(f"    -> Reduced Vector     : Dim {len(v_red)}, Min: {v_red.min():.4f}, Max: {v_red.max():.4f}, Mean: {v_red.mean():.4f}")
                
                journey_path = os.path.join("results/charts/query", f"{batch_name}_{t_name}_journey.png")
                chart_gen.generate_pipeline_journey_chart(j, journey_path)
                print(f"    -> Pipeline Journey Chart saved to: {journey_path}")
                
                if 'query_info' not in locals():
                    query_info = {
                        "query_image": out["query_image"],
                        "query_image_path": out["query_image_path"],
                        "query_label": out["query_label"]
                    }
                    pipeline_results_dict = {}
                    
                pipeline_results_dict[t_name] = {
                    "matches": out["results"],
                    "vector_stats": {
                        "raw_feature_vector": {
                            "dim": len(v_raw),
                            "min": float(v_raw.min()),
                            "max": float(v_raw.max()),
                            "mean": float(v_raw.mean())
                        },
                        "reduced_vector": {
                            "dim": len(v_red),
                            "min": float(v_red.min()),
                            "max": float(v_red.max()),
                            "mean": float(v_red.mean())
                        }
                    },
                    "journey_chart_path": journey_path
                }
                print(f"    Execution finished for {t_name}. Results captured.")
                
            else:
                print(f"\n[+] Running pipeline ({t_name})...")
                out_csv = os.path.join("results/pipelines", f"{t_name}.csv")
                out_json = os.path.join("results/pipelines", f"{t_name}.json")
                
                out = runner.run(gallery, probe, test_name=t_name)
                
                print(f"\n[+] Execution finished! Results saved to {out_csv}")
                print(f"    Timers: {out['timers']}")
                
                chart_gen.generate_all_charts(out_csv, out_json, t_name)
                
                csv_paths.append(out_csv)
                json_paths.append(out_json)
                test_names.append(t_name)
            
    if is_single_query and pipelines_to_run:
        import json
        unified_json_path = os.path.join("results", "query", f"{batch_name}.json")
        with open(unified_json_path, "w") as f:
            json.dump({
                "query_info": query_info,
                "pipeline_results": pipeline_results_dict
            }, f, indent=4)
        print(f"\n[+] Saved unified query results to {unified_json_path}")
        
        print(f"[+] Generating visual query report for {batch_name}...")
        chart_gen.generate_single_query_chart(batch_name, query_info, pipeline_results_dict)
        print(f"    Chart saved to {chart_gen.query_dir}/{batch_name}_query_results.png")
            
    elif not is_reduction_analysis and len(pipelines_to_run) > 1:
        chart_gen.generate_comparative_charts(batch_name, test_names, csv_paths, json_paths)

if __name__ == "__main__":
    run_tui()
