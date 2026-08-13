import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

class InformationLossDiagnostics:
    """
    Independent script to evaluate the impact of dimensionality reduction
    on the original vectors.
    """
    
    @staticmethod
    def mantle_test(original_vectors: np.ndarray, reduced_vectors: np.ndarray, 
                    metric: str = 'euclidean', correlation_type: str = 'pearson') -> float:
        """
        Calculates the correlation between the distance matrices of original 
        and reduced vectors.
        """
        # pdist returns condensed distance matrix (1D)
        dist_orig = pdist(original_vectors, metric=metric)
        dist_red = pdist(reduced_vectors, metric=metric)
        
        if correlation_type == 'pearson':
            corr, _ = pearsonr(dist_orig, dist_red)
        elif correlation_type == 'spearman':
            corr, _ = spearmanr(dist_orig, dist_red)
        else:
            raise ValueError("correlation_type must be 'pearson' or 'spearman'")
            
        return float(corr)

    @staticmethod
    def reconstruction_stress(original_vectors: np.ndarray, reconstructed_vectors: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error between the original and reconstructed data.
        Only applicable for reducers that support inverse_transform (like PCA, SVD).
        """
        if original_vectors.shape != reconstructed_vectors.shape:
            raise ValueError("Shapes of original and reconstructed vectors must match")
            
        mse = np.mean((original_vectors - reconstructed_vectors) ** 2)
        return float(mse)

    @staticmethod
    def trustworthiness(original_vectors: np.ndarray, reduced_vectors: np.ndarray, 
                        n_neighbors: int = 10, metric: str = 'euclidean') -> float:
        """
        Measures the extent to which the local structure is retained.
        """
        n_samples = original_vectors.shape[0]
        
        # Get nearest neighbors in original space
        nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
        nn_orig.fit(original_vectors)
        # Using [:, 1:] to exclude the point itself
        orig_indices = nn_orig.kneighbors(original_vectors, return_distance=False)[:, 1:]
        
        # Get nearest neighbors in reduced space
        nn_red = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
        nn_red.fit(reduced_vectors)
        red_indices = nn_red.kneighbors(reduced_vectors, return_distance=False)[:, 1:]
        
        # Calculate Trustworthiness
        # (This is a simplified rank-based overlap for neighborhood preservation)
        # We check how many of the true k nearest neighbors are retained in the reduced space
        retained = 0
        total = n_samples * n_neighbors
        
        for i in range(n_samples):
            # Intersection of neighbors
            overlap = len(set(orig_indices[i]).intersection(set(red_indices[i])))
            retained += overlap
            
        return retained / total
