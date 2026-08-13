import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add src to the Python path so it works seamlessly from the IDE
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Extractors
from core.extractors.cnn import ResNet50Extractor
from core.extractors.vit import DINOv2Extractor
from core.extractors.clip import CLIPExtractor

# Reducers
from core.reducers.pca import PCAReducer, SVDReducer
from core.reducers.umap_proj import UMAPReducer
from core.reducers.passthrough import PassThroughReducer

# Clusterers
from core.clustering.kmeans import MiniBatchKMeansClusterer
from core.clustering.hdbscan import HDBSCANClusterer
from core.clustering.graph import LouvainClusterer

# Search & Runner
from core.index.searcher import IndexSearcher
from core.runner import PipelineRunner
from core.data.loader import BivTattDatasetLoader, DatasetSplitter

def create_dummy_dataset(output_dir="dummy_dataset", n_gallery=20, n_probe=5):
    """Creates a dummy BivTatt dataset to test the pipelines."""
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    gallery = []
    probe = []
    
    for i in range(n_gallery):
        img_path = f"{output_dir}/images/lbl{i}_{i}.jpg"
        # Generate random image
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

def main():
    parser = argparse.ArgumentParser(description="Run ForensTatt Core CBIR Pipelines")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to the real local dataset directory")
    parser.add_argument("--kaggle-dataset", type=str, default="mexwell/bivtatt-dataset", help="Kaggle dataset handle to download and run automatically")
    parser.add_argument("--dummy", action="store_true", help="Force the use of a generated dummy dataset for testing")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., 'cpu', 'cuda'). Defaults to 'cuda' if available.")
    args = parser.parse_args()

    dataset_path = None
    
    if args.dummy:
        print("Creating dummy dataset (Force dummy mode)...")
        gallery, probe = create_dummy_dataset()
    else:
        if args.data_dir and os.path.exists(args.data_dir):
            dataset_path = args.data_dir
            print(f"Using local dataset from {dataset_path}...")
        elif args.kaggle_dataset:
            print(f"Downloading/Using Kaggle dataset '{args.kaggle_dataset}' via kagglehub...")
            import kagglehub
            # Automatically downloads or uses cached version
            dataset_path = kagglehub.dataset_download(args.kaggle_dataset)
            
            # Sometimes Kaggle zips have an extra root folder (e.g. BIVTatt-Dataset-master)
            if not os.path.exists(os.path.join(dataset_path, "images")):
                subdirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d))]
                for subdir in subdirs:
                    if os.path.exists(os.path.join(subdir, "images")):
                        dataset_path = subdir
                        break
                        
            print(f"Dataset resolved at: {dataset_path}")
            
        if dataset_path:
            loader = BivTattDatasetLoader(dataset_path)
            dataset = loader.load_dataset()
            splitter = DatasetSplitter()
            gallery, probe = splitter.split(dataset)
            print(f"Loaded {len(gallery)} gallery images and {len(probe)} probe images.")
        else:
            print("No real data_dir or kaggle_dataset provided. Falling back to dummy dataset...")
            gallery, probe = create_dummy_dataset()
    
    # ---------------------------------------------------------
    # Pipeline 1: ResNet50 | PCA | Euclidean | Minibatch-KMeans
    # ---------------------------------------------------------
    print("\n--- Running Pipeline 1 ---")
    p1_extractor = ResNet50Extractor(use_l2_norm=False, device=args.device)
    p1_reducer = PCAReducer(n_components=min(128, max(1, len(gallery) - 2)))
    p1_clusterer = MiniBatchKMeansClusterer(n_clusters=8, metric='euclidean')
    p1_searcher = IndexSearcher(metric='euclidean')
    
    runner1 = PipelineRunner(p1_extractor, p1_reducer, p1_clusterer, p1_searcher)
    out1 = runner1.run(gallery, probe, output_csv="results/pipeline1.csv", file_exists_action="overwrite")
    print(f"Pipeline 1 Timers: {out1['timers']}")
    
    # ---------------------------------------------------------
    # Pipeline 2: ViT/DINOv2 | UMAP | Euclidean | HDBSCAN
    # ---------------------------------------------------------
    print("\n--- Running Pipeline 2 ---")
    p2_extractor = DINOv2Extractor(device=args.device)
    p2_reducer = UMAPReducer(n_components=min(128, max(1, len(gallery) - 2)), metric='euclidean', n_neighbors=15)
    p2_clusterer = HDBSCANClusterer(min_cluster_size=5, metric='euclidean')
    p2_searcher = IndexSearcher(metric='euclidean')
    
    runner2 = PipelineRunner(p2_extractor, p2_reducer, p2_clusterer, p2_searcher)
    out2 = runner2.run(gallery, probe, output_csv="results/pipeline2.csv", file_exists_action="overwrite")
    print(f"Pipeline 2 Timers: {out2['timers']}")
    
    # ---------------------------------------------------------
    # Pipeline 3: CLIP | KNN Graph (Pass) | Cosine | Louvain
    # ---------------------------------------------------------
    print("\n--- Running Pipeline 3 ---")
    p3_extractor = CLIPExtractor(device=args.device)
    p3_reducer = PassThroughReducer() 
    p3_clusterer = LouvainClusterer(n_neighbors=5, metric='cosine')
    p3_searcher = IndexSearcher(metric='cosine')
    
    runner3 = PipelineRunner(p3_extractor, p3_reducer, p3_clusterer, p3_searcher)
    out3 = runner3.run(gallery, probe, output_csv="results/pipeline3.csv", file_exists_action="overwrite")
    print(f"Pipeline 3 Timers: {out3['timers']}")
    
    # ---------------------------------------------------------
    # Pipeline 4: Any(ResNet50) | TruncatedSVD | Dot Product | MiniBatch K-Means
    # ---------------------------------------------------------
    print("\n--- Running Pipeline 4 ---")
    p4_extractor = ResNet50Extractor(use_l2_norm=False, device=args.device)
    p4_reducer = SVDReducer(n_components=min(128, len(gallery)))
    p4_clusterer = MiniBatchKMeansClusterer(n_clusters=8, metric='dot')
    p4_searcher = IndexSearcher(metric='cosine') 
    
    runner4 = PipelineRunner(p4_extractor, p4_reducer, p4_clusterer, p4_searcher)
    out4 = runner4.run(gallery, probe, output_csv="results/pipeline4.csv", file_exists_action="overwrite")
    print(f"Pipeline 4 Timers: {out4['timers']}")

    print("\nAll pipelines executed successfully! Results saved in results/")

if __name__ == "__main__":
    main()
