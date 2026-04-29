import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from models.model_loader import load_al_model
from .dataloader import prepare_dataloader
from .utils import get_last_layer, free_gpu_mem, free_gpu_mem_gc, distance_torch, distance_np, write_selections
from nextFrag.config import get_project_root, DATASET_CONFIG

def diversity_al(
    dataset: str,
    arch: str,
    strategy: str,
    round_num: int,
    seed: int,
    num_selected: int=20_000,
    num_pca_components: int=64,
    batch_size: int=2048
):
    strategy=strategy.lower()
    if strategy != 'kmeans' and strategy != 'lcmd':
        raise ValueError(f"Only k-means and LCMD are supported. Received {strategy}.")

    data_path = get_project_root() / dataset / f'round_{round_num}' / strategy / f'{arch}_{seed}' / 'data' / 'pool.txt'
    seqsize = DATASET_CONFIG[dataset]['seqsize']

    dataloader = prepare_dataloader(
        data_path, 
        seqsize=seqsize,
        dset=dataset,
        batch_size=batch_size,
        shuffle = False,
        revcomp_same_batch = True
    )

    model=load_al_model(dataset=dataset,al_strategy=strategy,arch=arch,seed=seed,round_num=round_num-1)

    post_pca=IPCA(model=model,
                  dataloader=dataloader, 
                  n_components=num_pca_components,
                  batch_size=batch_size//2)
    
    if strategy == 'kmeans':
        selected_idx = _kmeans(post_pca, num_selected=num_selected)
    else: # lcmd
        selected_idx = LCMD(post_pca,num_clusters=num_selected)

    df=pd.read_csv(data_path,sep='\t',header=None)
    df=df.iloc[selected_idx]
    write_selections(
        df, 
        dataset=dataset,
        strategy=strategy,
        round_num=round_num,
        num_selected=num_selected,
        arch=arch,
        seed=seed
    )

def IPCA(model: nn.Module, 
         dataloader: DataLoader, 
         n_components: int, 
         batch_size: int=4096): 
    if torch.cuda.is_available():
        gpu = True
        device=torch.device("cuda")
        import cupy as cp
        from cuml.decomposition import IncrementalPCA
    else:
        gpu = False
        device=torch.device("cpu")
        from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=n_components, whiten=True, batch_size=batch_size)
    for batch in get_last_layer(model=model,
                                dataloader=dataloader,
                                device=device):
        if batch.shape[0] >= n_components:
            ipca.partial_fit(batch)
            if gpu:
                free_gpu_mem()
    
    n_samples=len(dataloader.dataset)//2
    results=cp.empty((n_samples,n_components)) if gpu else np.empty((n_samples,n_components))
    for i, batch in enumerate(get_last_layer(model=model,
                                             dataloader=dataloader, 
                                             device=device)):
        results[i*batch_size:min((i+1)*batch_size, n_samples),:]=ipca.transform(batch)
    
    if gpu:
        results=cp.asnumpy(results)

    del ipca
    if gpu:
        free_gpu_mem_gc()
        
    return results

def _kmeans(data: np.ndarray, num_selected: int) -> np.ndarray:
    if torch.cuda.is_available():
        from cuml.cluster import KMeans
        kmeans = KMeans(n_clusters=num_selected)
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
    else:
        kmeans = MiniBatchKMeans(n_clusters=num_selected)
        kmeans.fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

    selected_idx = np.zeros(centers.shape[0])
    for cluster_id in range(centers.shape[0]):
        mask = labels == cluster_id
        cluster_points = data[mask]

        if cluster_points.shape[0] > 0:
            distances = distance_np(centers[cluster_id],cluster_points)
            min_local_idx = np.argmin(distances)
            global_idx = np.arange(data.shape[0])[mask]
            selected_idx[cluster_id] = global_idx[min_local_idx]
    return selected_idx

def LCMD(data: np.ndarray,num_clusters: int, force_cpu: bool=False) -> np.ndarray:
    device = 'gpu' if torch.cuda.is_available() and not force_cpu else 'cpu'
    n_points, n_dims = data.shape
    
    data = torch.from_numpy(data).to(device=device)

    centers_idx = torch.empty(num_clusters, dtype=torch.int32,device=device)
    distances = torch.full((n_points,),float('inf'),device=device)
    closest_center=torch.zeros(n_points,dtype=torch.int32,device=device)

    # init
    idx1 = torch.randint(0,n_points,(1,),device=device)
    centers_idx[0]=idx1
    distances = distance_torch(data[idx1],data)

    #select another center
    centers_idx[1]=torch.argmax(distances)
    
    for idx in tqdm(range(1,num_clusters-1)):
        new_center = data[centers_idx[idx]]

        #calculate distance to new center
        distances_new = distance_torch(new_center,data)
        
        mask = distances_new < distances
        distances[mask] = distances_new[mask]
        closest_center[mask]=idx

        #find largest cluster
        cluster_sizes = torch.bincount(closest_center, weights=distances)
        largest_cluster_id = torch.argmax(cluster_sizes)
        
        #find new center
        mask2 = (closest_center == largest_cluster_id)
        cluster_idx = torch.where(mask2)[0]
        
        centers_idx[idx+1]=cluster_idx[torch.argmax(distances[mask2])]

    #return center idx
    return centers_idx.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("strategy",type=str,choices=['kmeans','lcmd'])
    parser.add_argument("round",type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--num_selected",type=int,default=20_000)
    parser.add_argument("--num_pca_components",type=int,default=64)
    parser.add_argument("--batch_size",type=int,default=4096)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")
        
    diversity_al(
        dataset=args.dataset,
        arch=args.arch,
        strategy=args.strategy,
        round_num=args.round,
        seed=args.seed,
        num_selected=args.num_selected,
        num_pca_components=args.num_pca_components,
        batch_size=args.batch_size
    )
    print("\nSelection complete!")

if __name__ == "__main__":
    main()