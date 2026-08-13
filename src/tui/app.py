import os
import uuid
import sys
import cv2
import numpy as np

try:
    import questionary
except ImportError:
    print("Please install questionary using 'uv add questionary'")
    sys.exit(1)

from core.data.loader import BivTattDatasetLoader, DatasetSplitter
from core.runner import PipelineRunner
from core.reduction_analyzer import ReductionAnalyzerRunner

from core.extractors.cnn import ResNet50Extractor
from core.extractors.vit import DINOv2Extractor
from core.extractors.clip import CLIPExtractor

from core.reducers.pca import PCAReducer, SVDReducer
from core.reducers.umap_proj import UMAPReducer
from core.reducers.passthrough import PassThroughReducer

from core.clustering.kmeans import MiniBatchKMeansClusterer
from core.clustering.hdbscan import HDBSCANClusterer
from core.clustering.graph import LouvainClusterer

from core.index.searcher import IndexSearcher

EXTRACTORS = {
    "ResNet50": ResNet50Extractor,
    "DINOv2": DINOv2Extractor,
    "CLIP": CLIPExtractor
}

REDUCERS = {
    "PCA": PCAReducer,
    "SVD": SVDReducer,
    "UMAP": UMAPReducer,
    "PassThrough": PassThroughReducer
}

CLUSTERERS = {
    "MiniBatchKMeans": MiniBatchKMeansClusterer,
    "HDBSCAN": HDBSCANClusterer,
    "Louvain": LouvainClusterer
}

SEARCHERS = {
    "IndexSearcher (Euclidean)": lambda: IndexSearcher(metric='euclidean'),
    "IndexSearcher (Cosine)": lambda: IndexSearcher(metric='cosine')
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
    print(" 🔎 ForensTatt Core - Interactive Pipeline Builder")
    print("=" * 60)
    
    mode = questionary.select(
        "Select the mode of operation:",
        choices=[
            "1. Full Pipeline Execution",
            "2. Dimensionality Reduction Analysis"
        ]
    ).ask()
    
    if not mode:
        return
        
    is_reduction_analysis = "2." in mode

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
    
    # 2. Partitioning
    start_str = questionary.text("Dataset partition start index (default 0):", validate=is_valid_int).ask()
    if start_str is None: return
    length_str = questionary.text("Dataset partition length (default all):", validate=is_valid_int).ask()
    if length_str is None: return
    
    start_idx = int(start_str) if start_str.strip() else 0
    length_idx = int(length_str) if length_str.strip() else None

    # Load Data
    if "Dummy" in dataset_source:
        print("Creating dummy dataset...")
        g, p = create_dummy_dataset()
        dataset = g + p
        if length_idx is not None:
            dataset = dataset[start_idx:start_idx+length_idx]
        else:
            dataset = dataset[start_idx:]
            
        dataset_full = dataset
        splitter = DatasetSplitter()
        gallery, probe = splitter.split(dataset)
        print(f"Loaded {len(gallery)} gallery images and {len(probe)} probe images.")
    else:
        if "Local" in dataset_source:
            dataset_path = questionary.path("Enter path to the local BivTatt directory:").ask()
            if not dataset_path: return
            
        elif "Kagglehub" in dataset_source:
            print("Resolving dataset via Kagglehub...")
            import kagglehub
            # Change kagglehub cache directory to our data folder if possible
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
            dataset = loader.load_dataset(start=start_idx, length=length_idx)
            dataset_full = dataset
            splitter = DatasetSplitter()
            gallery, probe = splitter.split(dataset)
            print(f"Loaded {len(gallery)} gallery images and {len(probe)} probe images.")

    if not gallery or not probe:
        print("Dataset does not have enough gallery/probe splits. Exiting.")
        return

    # 3. Component Selection
    device = questionary.select("Select computation device:", choices=["cuda", "cpu"]).ask()
    if not device: return
    
    extractor_choice = questionary.select("Select Feature Extractor:", choices=list(EXTRACTORS.keys())).ask()
    if not extractor_choice: return
    
    reducer_choice = questionary.select("Select Dimensionality Reducer:", choices=list(REDUCERS.keys())).ask()
    if not reducer_choice: return
    
    clusterer_choice = None
    searcher_choice = None
    
    if not is_reduction_analysis:
        clusterer_choice = questionary.select("Select Clusterer:", choices=list(CLUSTERERS.keys())).ask()
        if not clusterer_choice: return
        
        searcher_choice = questionary.select("Select Index Searcher:", choices=list(SEARCHERS.keys())).ask()
        if not searcher_choice: return

    # 4. Confirmation
    confirm = questionary.confirm("Execute?").ask()
    if not confirm:
        return

    print("\n[+] Building components...")
    if extractor_choice == "ResNet50":
        extractor = EXTRACTORS[extractor_choice](use_l2_norm=False, device=device)
    else:
        extractor = EXTRACTORS[extractor_choice](device=device)
    
    # Reducers dynamic setup
    red_k = min(128, max(1, len(gallery) - 2))
    if reducer_choice in ["PCA", "SVD"]:
        reducer = REDUCERS[reducer_choice](n_components=red_k)
    elif reducer_choice == "UMAP":
        reducer = REDUCERS[reducer_choice](n_components=red_k, metric='euclidean', n_neighbors=15)
    else:
        reducer = REDUCERS[reducer_choice]()

    if is_reduction_analysis:
        output_json = f"results/reduction-analysis/reduction_{uuid.uuid4().hex[:8]}.json"
        print("\n[+] Running Reduction Analysis...")
        runner = ReductionAnalyzerRunner(extractor, reducer)
        # We can use the gallery to test reduction or the full dataset. Full is better to see variance.
        out = runner.run(dataset_full, output_json=output_json)
        
        print(f"\n[+] Execution finished! Results saved to {output_json}")
        print("-" * 40)
        print(f"Information Retained: {out['information_retained_percent']:.2f}%")
        print(f"Information Lost:     {out['information_lost_percent']:.2f}%")
        if 'explained_variance_percent' in out:
            print(f"Explained Variance:   {out['explained_variance_percent']:.2f}%")
        print("-" * 40)
        
    else:
        # Clusterers dynamic setup
        if clusterer_choice == "MiniBatchKMeans":
            clusterer = CLUSTERERS[clusterer_choice](n_clusters=8, metric='euclidean')
        elif clusterer_choice == "HDBSCAN":
            clusterer = CLUSTERERS[clusterer_choice](min_cluster_size=max(2, min(5, len(gallery)//2)), metric='euclidean')
        elif clusterer_choice == "Louvain":
            clusterer = CLUSTERERS[clusterer_choice](n_neighbors=5, metric='cosine')

        searcher = SEARCHERS[searcher_choice]()

        output_csv = f"results/pipelines/tui_pipeline_{uuid.uuid4().hex[:8]}.csv"

        print("\n[+] Running pipeline...")
        runner = PipelineRunner(extractor, reducer, clusterer, searcher)
        out = runner.run(gallery, probe, output_csv=output_csv, file_exists_action="overwrite")
        
        print(f"\n[+] Execution finished! Results saved to {output_csv}")
        print(f"    Timers: {out['timers']}")

if __name__ == "__main__":
    run_tui()
