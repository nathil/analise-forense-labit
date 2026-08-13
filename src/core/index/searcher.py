import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import pairwise_distances

class IndexSearcher:
    """
    Organizes the gallery vectors partitioned by clusters.
    On query, filters by comparing with centroids first, 
    and then scanning only the nearest cluster.
    """
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self.cluster_centers = None
        self.cluster_vectors = {}  # cluster_id -> vectors
        self.cluster_metadata = {} # cluster_id -> list of gallery indices/metadata
        self.is_indexed = False

    def build_index(self, gallery_vectors: np.ndarray, 
                    gallery_metadata: List[Dict],
                    cluster_centers: np.ndarray, 
                    cluster_labels: np.ndarray) -> None:
        """
        Builds the partition index based on cluster assignments.
        """
        self.cluster_centers = cluster_centers
        n_clusters = cluster_centers.shape[0]
        
        for i in range(n_clusters):
            idx = np.where(cluster_labels == i)[0]
            if len(idx) > 0:
                self.cluster_vectors[i] = gallery_vectors[idx]
                self.cluster_metadata[i] = [gallery_metadata[j] for j in idx]
                
        self.is_indexed = True

    def search(self, probe_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Searches the index for the closest matches to the probe vector.
        Returns a list of tuples (metadata, distance).
        """
        if not self.is_indexed:
            raise ValueError("Index has not been built yet.")
            
        probe_2d = probe_vector.reshape(1, -1)
        
        # 1. Find closest centroid
        centroid_dists = pairwise_distances(probe_2d, self.cluster_centers, metric=self.metric)[0]
        best_cluster = np.argmin(centroid_dists)
        
        # 2. Search only within the closest cluster
        cluster_vecs = self.cluster_vectors.get(best_cluster)
        cluster_meta = self.cluster_metadata.get(best_cluster)
        
        if cluster_vecs is None or len(cluster_vecs) == 0:
            return []
            
        dists = pairwise_distances(probe_2d, cluster_vecs, metric=self.metric)[0]
        
        # 3. Get Top K
        top_indices = np.argsort(dists)[:top_k]
        
        results = []
        for i in top_indices:
            results.append((cluster_meta[i], float(dists[i])))
            
        return results
