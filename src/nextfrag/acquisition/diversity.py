import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from nextfrag.models.model_loader import load_model
from nextfrag.data.dataloader import build_dataloader
from nextfrag.utils import write_selections, free_gpu_mem, free_gpu_mem_gc
from nextfrag.config import DATASET_CONFIG, DEFAULT_N_SELECTED
from nextfrag.path_resolver import PathResolver
from .utils import get_last_layer, distance_torch, distance_np


def diversity_al(
    cfg: PathResolver,
    n_pca_components: int = 64,
    batch_size: int = 2048,
):
    """Select sequences by diversity in model embedding space (k-means or LCMD).

    Extracts penultimate-layer features via incremental PCA, then applies the
    strategy named in cfg.acquisition ('kmeans' or 'lcmd') to pick n_selected
    sequences.

    Args:
        cfg: PathResolver for the current round. cfg.acquisition must be
             'kmeans' or 'lcmd'.
        n_pca_components: Number of PCA components to retain before clustering.
        batch_size: Batch size for feature extraction.
    """
    strategy = cfg.acquisition.lower()
    if strategy not in ('kmeans', 'lcmd'):
        raise ValueError(f"Only k-means and LCMD are supported. Received {strategy}.")

    prev = cfg.prev_round()
    seqsize = DATASET_CONFIG[cfg.dataset]['seqsize']

    dataloader = build_dataloader(
        prev.pool_path,
        seqsize=seqsize,
        dataset=cfg.dataset,
        batch_size=batch_size,
        shuffle=False,
        revcomp_same_batch=True,
    )

    model = load_model(path=prev.model_path, dataset=cfg.dataset, arch=cfg.arch)

    post_pca = IPCA(model=model, dataloader=dataloader,
                    n_components=n_pca_components, batch_size=batch_size // 2)

    if strategy == 'kmeans':
        selected_idx = _kmeans(post_pca, n_selected=cfg.n_selected)
    else:
        selected_idx = LCMD(post_pca, n_clusters=cfg.n_selected)

    df = pd.read_csv(prev.pool_path, sep='\t', header=None)
    df = df.iloc[selected_idx]
    write_selections(cfg, result_df=df)


def IPCA(model: nn.Module,
         dataloader: DataLoader,
         n_components: int,
         batch_size: int = 4096) -> np.ndarray:
    """Extract whitened PCA features from the model's penultimate layer.

    Args:
        model: Trained model with a ``final_block`` attribute.
        dataloader: Pool dataloader with revcomp_same_batch=True.
        n_components: Number of PCA components to retain.
        batch_size: Number of sequences per PCA mini-batch.

    Returns:
        Array of shape (n_pool_seqs, n_components).
    """
    if torch.cuda.is_available():
        gpu = True
        device = torch.device("cuda")
        import cupy as cp
        from cuml.decomposition import IncrementalPCA
    else:
        gpu = False
        device = torch.device("cpu")
        from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=n_components, whiten=True, batch_size=batch_size)
    for batch in get_last_layer(model=model, dataloader=dataloader, device=device):
        if batch.shape[0] >= n_components:
            if gpu:
                ipca.partial_fit(cp.from_dlpack(batch.detach()))
                free_gpu_mem()
            else:
                ipca.partial_fit(batch.detach().numpy())

    n_samples = len(dataloader.dataset) // 2
    results = cp.empty((n_samples, n_components)) if gpu else np.empty((n_samples, n_components))
    for i, batch in enumerate(get_last_layer(model=model, dataloader=dataloader, device=device)):
        if gpu:
            results[i * batch_size:min((i + 1) * batch_size, n_samples), :] = ipca.transform(cp.from_dlpack(batch.detach()))
        else:
            results[i * batch_size:min((i + 1) * batch_size, n_samples), :] = ipca.transform(batch.detach().numpy())

    if gpu:
        results = cp.asnumpy(results)

    del ipca
    if gpu:
        free_gpu_mem_gc()

    return results


def _kmeans(data: np.ndarray, n_selected: int) -> np.ndarray:
    """Select one sequence per k-means cluster, choosing the point nearest to each centroid.

    Uses cuML KMeans on GPU and scikit-learn MiniBatchKMeans on CPU.

    Args:
        data: Feature matrix of shape (n_pool_seqs, n_features).
        n_selected: Number of clusters (= number of sequences to select).

    Returns:
        Array of shape (n_selected,) containing selected pool indices.
    """
    if torch.cuda.is_available():
        from cuml.cluster import KMeans
        kmeans = KMeans(n_clusters=n_selected)
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_selected)
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

    selected_idx = np.zeros(centers.shape[0])
    for cluster_id in range(centers.shape[0]):
        mask = labels == cluster_id
        cluster_points = data[mask]
        if cluster_points.shape[0] > 0:
            distances = distance_np(centers[cluster_id], cluster_points)
            min_local_idx = np.argmin(distances)
            global_idx = np.arange(data.shape[0])[mask]
            selected_idx[cluster_id] = global_idx[min_local_idx]
    return selected_idx


def LCMD(data: np.ndarray, n_clusters: int, force_cpu: bool = False) -> np.ndarray:
    """Largest Cluster Maximum Distance selection.

    Greedily selects n_clusters points that are spread across the feature space.

    Args:
        data: Feature matrix of shape (n_pool_seqs, n_features).
        n_clusters: Number of sequences to select.
        force_cpu: If True, run on CPU even when a GPU is available.

    Returns:
        Array of shape (n_clusters,) containing selected pool indices.
    """
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    n_points, n_dims = data.shape

    data = torch.from_numpy(data).to(device=device)

    centers_idx = torch.empty(n_clusters, dtype=torch.int32, device=device)
    distances = torch.full((n_points,), float('inf'), device=device)
    closest_center = torch.zeros(n_points, dtype=torch.int32, device=device)

    idx1 = torch.randint(0, n_points, (1,), device=device)
    centers_idx[0] = idx1
    distances = distance_torch(data[idx1], data)
    centers_idx[1] = torch.argmax(distances)

    for idx in tqdm(range(1, n_clusters - 1)):
        new_center = data[centers_idx[idx]]
        distances_new = distance_torch(new_center, data)
        mask = distances_new < distances
        distances[mask] = distances_new[mask]
        closest_center[mask] = idx

        cluster_sizes = torch.bincount(closest_center, weights=distances)
        largest_cluster_id = torch.argmax(cluster_sizes)
        mask2 = (closest_center == largest_cluster_id)
        cluster_idx = torch.where(mask2)[0]
        centers_idx[idx + 1] = cluster_idx[torch.argmax(distances[mask2])]

    return centers_idx.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['yeast', 'human'], required=True)
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--acquisition", choices=['kmeans', 'lcmd'], required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-selected", type=int, default=DEFAULT_N_SELECTED)
    parser.add_argument("--n-pca-components", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")

    cfg = PathResolver(
        dataset=args.dataset,
        round_num=args.round,
        arch=args.arch,
        seed=args.seed,
        acquisition=args.acquisition,
        n_selected=args.n_selected,
    )
    diversity_al(cfg, n_pca_components=args.n_pca_components, batch_size=args.batch_size)
    print("\nSelection complete!")


if __name__ == "__main__":
    main()
