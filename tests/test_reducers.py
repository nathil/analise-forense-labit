import pytest
import numpy as np

from core.reducers.pca import PCAReducer, SVDReducer
from core.reducers.umap_proj import RandomProjectionReducer

@pytest.fixture
def dummy_vectors():
    # 150 vectors, 2048 dimensions (needs to be > 128 for PCA/SVD)
    return np.random.rand(150, 2048)

def test_pca_reducer_shape(dummy_vectors):
    reducer = PCAReducer(n_components=128)
    reducer.fit(dummy_vectors)
    reduced = reducer.transform(dummy_vectors)
    
    assert reduced.shape == (150, 128), "PCA should reduce dimensions to 128"

def test_svd_reducer_shape(dummy_vectors):
    reducer = SVDReducer(n_components=64)
    reducer.fit(dummy_vectors)
    reduced = reducer.transform(dummy_vectors)
    
    assert reduced.shape == (150, 64), "SVD should reduce dimensions to 64"

def test_random_projection_shape(dummy_vectors):
    reducer = RandomProjectionReducer(n_components=256)
    reducer.fit(dummy_vectors)
    reduced = reducer.transform(dummy_vectors)
    
    assert reduced.shape == (150, 256), "Random Projection should reduce dimensions to 256"
