# ForensTatt Core - Project Guidelines

1. **Language & Environment**: 
   - Language: Python.
   - All code, documentation, and inline comments MUST be in English.
   - Package Manager: `uv`.

2. **Git & Environments**:
   - Maintain a clean Git history with atomic, well-defined commits for each logical component.
   - Every time an environment variable is added or modified, the `.env.example` file MUST be updated simultaneously.

3. **Output & Execution constraints**:
   - All test or execution results must be saved in the `results/` directory.
   - Output files will be in CSV format.
   - If an output filename already exists, the script/API must provide an option to either `append` to it or `overwrite` it.
   - No CLI tool will be built initially; focus exclusively on clean core classes and APIs.

4. **Architecture Blueprint**:
   - Follow the BivTatt Kaggle dataset protocol (Strict bounding box cropping, Gallery=originals, Probe=transformations).
   - Core layers: Feature Extractor (ResNet, DINOv2, CLIP), Dimensionality Reducer (PCA, SVD, UMAP, RandomProjection), Clusterer (KMeans with L2 for Cosine, Louvain via KNNGraph) & IndexSearcher.
   - Standalone Informational Loss Module for evaluating dimensionality reduction distortion (Mantle Test, Stress, Trustworthiness).
