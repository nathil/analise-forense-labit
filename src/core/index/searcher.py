import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import pairwise_distances

class IndexSearcher:
    """
    Organizes the gallery vectors partitioned by clusters.
    On query, filters by comparing with centroids first using Multi-Probe IVF
    (inspecting the n_probe closest clusters and expanding if needed to guarantee
    a rich candidate pool), and then ranks candidates to return the exact Top-K.
    """
    def __init__(self, metric: str = 'euclidean', n_probe: int = 5):
        self.metric = metric.lower()
        self.n_probe = max(1, n_probe)
        self.cluster_centers = None
        self.cluster_vectors = {}   # cluster_id -> vectors
        self.cluster_metadata = {}  # cluster_id -> list of gallery indices/metadata
        self.is_indexed = False

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if self.metric in ['cosine', 'dot']:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            return vectors / norms
        return vectors

    def build_index(self, gallery_vectors: np.ndarray, 
                    gallery_metadata: List[Dict],
                    cluster_centers: np.ndarray, 
                    cluster_labels: np.ndarray) -> None:
        """
        Builds the partition index based on cluster assignments.
        """
        self.cluster_vectors = {}
        self.cluster_metadata = {}
        
        norm_gallery = self._normalize(gallery_vectors)
        self.cluster_centers = self._normalize(cluster_centers)
        n_clusters = self.cluster_centers.shape[0]
        
        for i in range(n_clusters):
            idx = np.where(cluster_labels == i)[0]
            if len(idx) > 0:
                self.cluster_vectors[i] = norm_gallery[idx]
                self.cluster_metadata[i] = [gallery_metadata[j] for j in idx]
                
        # Fallback if any items had labels outside [0, n_clusters-1]
        if not self.cluster_vectors and len(gallery_vectors) > 0:
            self.cluster_centers = np.mean(norm_gallery, axis=0, keepdims=True)
            self.cluster_vectors[0] = norm_gallery
            self.cluster_metadata[0] = list(gallery_metadata)
                
        self.is_indexed = True

    def search(self, probe_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Searches the index for the closest matches to the probe vector using
        Multi-Probe cluster expansion.
        Returns a list of tuples (metadata, distance).
        """
        if not self.is_indexed:
            raise ValueError("Index has not been built yet.")
            
        probe_2d = self._normalize(probe_vector.reshape(1, -1))
        
        # 1. Rank all centroids by distance to probe
        if self.metric == 'dot':
            centroid_dists = -np.dot(probe_2d, self.cluster_centers.T)[0]
        else:
            centroid_dists = pairwise_distances(probe_2d, self.cluster_centers, metric=self.metric)[0]
            
        sorted_clusters = np.argsort(centroid_dists)
        
        # 2. Multi-Probe expansion: gather candidates from top n_probe clusters,
        # and keep expanding if total candidates < min_candidates (at least top_k * 4 or 150)
        min_candidates = max(top_k * 4, 150)
        candidate_vecs_list = []
        candidate_meta_list = []
        probed_count = 0
        
        for cid in sorted_clusters:
            c_vecs = self.cluster_vectors.get(int(cid))
            c_meta = self.cluster_metadata.get(int(cid))
            if c_vecs is None or len(c_vecs) == 0:
                continue
                
            candidate_vecs_list.append(c_vecs)
            candidate_meta_list.extend(c_meta)
            probed_count += 1
            
            if probed_count >= self.n_probe and len(candidate_meta_list) >= min_candidates:
                break
                
        if not candidate_vecs_list:
            return []
            
        pooled_vecs = np.vstack(candidate_vecs_list)
        
        # 3. Compute exact distances across pooled candidates
        if self.metric == 'dot':
            dists = -np.dot(probe_2d, pooled_vecs.T)[0]
        else:
            dists = pairwise_distances(probe_2d, pooled_vecs, metric=self.metric)[0]
        
        # 4. Get Top K
        top_indices = np.argsort(dists)[:top_k]
        
        results = []
        for i in top_indices:
            results.append((candidate_meta_list[i], float(dists[i])))
            
        return results
