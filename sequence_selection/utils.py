import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from cuml.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

def enable_dropout(model: nn.Module):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class LayerInputExtractor:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer
        self._features = None
        # register hook
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        # save the output of the chosen layer
        self._features = inputs[0].detach()

    def __call__(self, x):
        _ = self.model(x)
        return self._features

    def close(self):
        self.hook.remove()

def last_layer_features(dataloader: DataLoader,
                        model: nn.Module) -> np.array:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device).eval()
    extractor=LayerInputExtractor(model,model.final_linear[0])

    results=[]
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            results.append(extractor(X))
    combined = torch.cat(results).cpu().numpy()
    num_samples=combined.shape[0]//2
    return (combined[:num_samples]+combined[num_samples:])/2

def distance_np(target: np.array, X: np.array) -> np.array:
    return np.sum((target-X)**2,axis=1)

def distance_torch(target: torch.Tensor,X: torch.Tensor) -> torch.Tensor:
    return torch.sum((target-X)**2,axis=1)

def _kmeans(data: np.array, num_selected: int) -> np.array:
    if torch.cuda.is_available():
        kmeans = KMeans(n_clusters=num_selected)
        clustered=kmeans.fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
    else: 
        centers, labels, inertia = MiniBatchKMeans(data,n_clusters=num_selected)

    selected_idx = np.zeros(centers.shape[0])
    for cluster_id in range(centers.shape[0]):
        mask = labels == cluster_id
        cluster_points = data[mask]

        if cluster_points.shape[0] > 0:
            distances = distance_np(centers[cluster_id],cluster_points)
            min_index_local = np.argmin(distances)
            global_indices = np.arange(data.shape[0])[mask]
            selected_idx[cluster_id] = global_indices[min_index_local]
    return selected_idx

def LCMD_cpu(data: np.array, num_clusters: int) -> np.array:
    n_points, n_dims = data.shape
    
    centers_idx = np.empty(num_clusters, dtype=np.int32)
    distances = np.full(n_points,np.inf)
    closest_center=np.zeros(n_points,dtype=np.int32)

    # init
    idx1 = np.random.choice(n_points)
    centers_idx[0]=idx1
    distances = distance_np(data[idx1],data)

    #select another center
    centers_idx[1]=distances.argmax()
    
    for idx in tqdm(range(1,num_clusters-1)):
        new_center = data[centers_idx[idx]]

        #calculate distance to new center
        distances_new = distance_np(new_center,data)
        
        mask = distances_new < distances
        distances[mask] = distances_new[mask]
        closest_center[mask]=idx

        #find largest cluster
        largest_cluster_id = np.bincount(closest_center, weights=distances).argmax()
        
        #find new center
        mask2 = (closest_center == largest_cluster_id)
        cluster_idx = np.where(mask2)[0]
        
        centers_idx[idx+1]=cluster_idx[distances[mask2].argmax()]

    #return center idx
    return centers_idx

def LCMD_gpu(data: np.array,num_clusters: int) -> np.array:
    n_points, n_dims = data.shape
    
    data_gpu = torch.from_numpy(data).to(device='cuda')

    centers_idx = torch.empty(num_clusters, dtype=torch.int32,device='cuda')
    distances = torch.full((n_points,),float('inf'),device='cuda')
    closest_center=torch.zeros(n_points,dtype=torch.int32,device='cuda')

    # init
    idx1 = torch.randint(0,n_points,(1,),device='cuda')
    centers_idx[0]=idx1
    distances = distance_torch(data_gpu[idx1],data_gpu)

    #select another center
    centers_idx[1]=torch.argmax(distances)
    
    for idx in tqdm(range(1,num_clusters-1)):
        new_center = data_gpu[centers_idx[idx]]

        #calculate distance to new center
        distances_new = distance_torch(new_center,data_gpu)
        
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

def LCMD(data: np.array, num_clusters: int) -> np.array:
    if torch.cuda.is_available(): # run on gpu if possible
        return LCMD_gpu(data,num_clusters)
    else:
        return LCMD_cpu(data,num_clusters)