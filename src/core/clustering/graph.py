import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np

from .base import Clusterer

class LouvainClusterer(Clusterer):
    """
    Louvain community detection algorithm for clustering.
    Calculates a KNNGraph intermediate structure before running the detection.
    """
    def __init__(self, n_neighbors: int = 15, metric: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = 'cosine' if metric.lower() == 'dot' else metric.lower()
        self.labels_ = None
        self._cluster_centers = None

    def _preprocess(self, vectors: np.ndarray) -> np.ndarray:
        if self.metric == 'cosine':
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            return vectors / norms
        return vectors
        
    def fit(self, vectors: np.ndarray) -> None:
        """
        Builds a KNN graph and applies Louvain community detection.
        """
        proc_vectors = self._preprocess(vectors)
        n_samples = proc_vectors.shape[0]
        actual_neighbors = min(self.n_neighbors, max(1, n_samples - 1))
        
        # 1. Generate Adjacency Matrix (KNN Graph)
        nn = NearestNeighbors(n_neighbors=actual_neighbors, metric=self.metric)
        nn.fit(proc_vectors)
        # mode='distance' gives us distances, we want similarities for edge weights
        adj_matrix = nn.kneighbors_graph(proc_vectors, mode='distance')
        
        # Convert distances to weights (similarity = 1 / (1 + distance))
        # and create networkx graph
        G = nx.Graph()
        G.add_nodes_from(range(n_samples))
        
        # Iterate over non-zero elements in sparse matrix
        cx = adj_matrix.tocoo()
        for i, j, dist in zip(cx.row, cx.col, cx.data):
            if i != j:  # ignore self loops for community detection
                weight = 1.0 / (1.0 + max(0.0, float(dist)))
                G.add_edge(i, j, weight=weight)
                
        # 2. Apply Louvain algorithm
        try:
            # networkx >= 2.6 provides louvain_communities
            communities = nx.community.louvain_communities(G, weight='weight', seed=42)
            
            self.labels_ = np.zeros(n_samples, dtype=int)
            for cluster_idx, community in enumerate(communities):
                for node in community:
                    self.labels_[node] = cluster_idx
                    
            # 3. Calculate Centroids
            n_clusters = len(communities)
            self._cluster_centers = np.zeros((n_clusters, proc_vectors.shape[1]))
            for i in range(n_clusters):
                cluster_pts = proc_vectors[self.labels_ == i]
                if len(cluster_pts) > 0:
                    self._cluster_centers[i] = np.mean(cluster_pts, axis=0)
            self._cluster_centers = self._preprocess(self._cluster_centers)
                    
        except AttributeError:
            raise ImportError("networkx.community.louvain_communities requires networkx>=2.6")

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        For a new vector, Louvain doesn't have a strict predict.
        We approximate by assigning it to the closest centroid.
        """
        if self._cluster_centers is None:
            raise ValueError("Model is not fitted yet.")
            
        proc_vectors = self._preprocess(vectors)
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        nn.fit(self._cluster_centers)
        
        _, indices = nn.kneighbors(proc_vectors)
        return indices.flatten()
        
    @property
    def cluster_centers(self) -> np.ndarray:
        return self._cluster_centers
