import cuml
import argparse
import pandas as pd
from pathlib import Path
from .diversity_strategies import IPCA
from .dataloader import prepare_dataloader
from models.model_loader import load_model
from nextFrag.config import get_project_root, DATASET_CONFIG

def umap(
    dataset: str,
    model_path: str | Path,
    out_path: str | Path,
    data_path: str | Path = None,
    arch: str='rnn',
    num_pca_components: int=64,
    batch_size: int=2048
):
    if data_path is None:
        data_path = get_project_root() / dataset / 'round_0' / 'common' / 'common' / 'data' / 'pool.txt'
    seqsize = DATASET_CONFIG[dataset]['seqsize']

    dataloader = prepare_dataloader(
        data_path, 
        seqsize=seqsize,
        dset=dataset,
        batch_size=batch_size,
        shuffle = False,
        revcomp_same_batch = True
    )

    model=load_model(dataset=dataset,arch=arch,path=model_path)

    X=IPCA(model=model,
           dataloader=dataloader, 
           n_components=num_pca_components,
           batch_size=batch_size//2)
    reducer = cuml.manifold.UMAP(n_components=2, n_neighbors=15, n_epochs=200)
    embedding = reducer.fit_transform(X)

    df=pd.read_csv(data_path,sep='\t',header=None)
    df.columns=['seq','expr']
    df['umap0']=embedding[:,0]
    df['umap1']=embedding[:,1]
    df.to_csv(out_path,index=None,sep='\t')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=['yeast','human'])
    parser.add_argument("model_path",type=str)
    parser.add_argument("out_path",type=str)
    parser.add_argument("--data_path",type=str,default=None)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")
        
    umap(dataset=args.dataset,
         model_path=args.model_path, 
         out_path=args.out_path,
         data_path=args.data_path)

if __name__ == "__main__":
    main()