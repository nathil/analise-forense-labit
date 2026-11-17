import pytest
import numpy as np
import os
import cv2

from core.extractors.base import FeatureExtractor
from core.reducers.pca import PCAReducer
from core.clustering.kmeans import KMeansClusterer
from core.clustering.graph import LouvainClusterer
from core.index.searcher import IndexSearcher
from core.runner import PipelineRunner

class MockExtractor(FeatureExtractor):
    def extract(self, image):
        # Return random 2048 vector
        return np.random.rand(2048)

@pytest.fixture
def mock_pipeline_data(tmp_path):
    # Create dummy images
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    gallery = []
    probe = []
    
    # 10 gallery, 2 probe
    for i in range(10):
        img_path = str(img_dir / f"lbl{i}_{i}.jpg")
        cv2.imwrite(img_path, np.zeros((50, 50, 3), dtype=np.uint8))
        gallery.append({"label": f"lbl{i}", "filename": f"lbl{i}_{i}.jpg", "image_path": img_path, "box_path": None})
        
    for i in range(2):
        img_path = str(img_dir / f"lbl{i}_{i}_trans.jpg")
        cv2.imwrite(img_path, np.zeros((50, 50, 3), dtype=np.uint8))
        probe.append({"label": f"lbl{i}", "filename": f"lbl{i}_{i}_trans.jpg", "image_path": img_path, "box_path": None, "transformation": "trans"})
        
    return gallery, probe, str(tmp_path / "results.csv")

def test_louvain_graph_coupling():
    # Test that Louvain generates a valid graph and centers
    clusterer = LouvainClusterer(n_neighbors=2)
    vectors = np.random.rand(10, 128)
    clusterer.fit(vectors)
    
    assert clusterer.labels_ is not None, "Louvain should assign labels"
    assert clusterer.cluster_centers is not None, "Louvain should calculate centroids"
    
def test_pipeline_runner(mock_pipeline_data):
    gallery, probe, out_csv = mock_pipeline_data
    
    extractor = MockExtractor()
    reducer = PCAReducer(n_components=2)
    # Using KMeans for simplicity
    clusterer = KMeansClusterer(n_clusters=2)
    searcher = IndexSearcher()
    
    runner = PipelineRunner(extractor, reducer, clusterer, searcher)
    
    output = runner.run(gallery, probe, output_csv=out_csv, file_exists_action="overwrite", top_k=2)
    
    assert "timers" in output
    assert "results" in output
    
    # 2 probes * top_k=2 = 4 results
    assert len(output["results"]) == 4
    assert os.path.exists(out_csv), "Output CSV should be created"
    
    # Test append mode
    runner.run(gallery, probe, output_csv=out_csv, file_exists_action="append", top_k=2)
    # The file should now have 4 (header+4) + 4 = 8 lines (or similar pandas structure)
    with open(out_csv, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 9, "CSV should have 1 header line + 8 data lines after append"

def test_pipeline_runner_multi_topk(mock_pipeline_data, tmp_path):
    gallery, probe, _ = mock_pipeline_data
    extractor = MockExtractor()
    reducer = PCAReducer(n_components=2)
    clusterer = KMeansClusterer(n_clusters=2)
    searcher = IndexSearcher()

    runner = PipelineRunner(extractor, reducer, clusterer, searcher)
    out_dir = str(tmp_path / "multi_k")
    out = runner.run(
        gallery,
        probe,
        test_name="TestMultiK",
        output_dir=out_dir,
        file_exists_action="overwrite",
        top_k_list=[1, 5, 10, 20, 50]
    )

    assert "metrics_by_top_k" in out["metrics"]
    for k in [1, 5, 10, 20, 50]:
        assert str(k) in out["metrics"]["metrics_by_top_k"]
        assert os.path.exists(os.path.join(out_dir, f"top_k_{k}", "TestMultiK.json"))
        assert os.path.exists(os.path.join(out_dir, f"top_k_{k}", "TestMultiK.csv"))
