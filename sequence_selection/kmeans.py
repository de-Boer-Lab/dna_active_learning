import pandas as pd
import os, argparse
from models.model_utils import load_model
from .utils import last_layer_features, _kmeans
from models.dl_utils import prepare_dataloader

def kmeans_al(species: str,
             arch: str,
             round: int,
             seed: int,
             num_selected: int):
    
    data_path = f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}/round_{round-1}/kmeans/{arch}_{seed}/pool.txt"
    seqsize = 200 if species == 'human' else 150
    batch_size = 4096

    dataloader = prepare_dataloader(data_path, 
                                    seqsize=seqsize,
                                    species=species,
                                    batch_size=batch_size,
                                    shuffle = False)

    model=load_model(species=species,al_method='kmeans',arch=arch,seed=seed,round=round-1)

    last_layer=last_layer_features(dataloader,model=model)
    selected_idx = _kmeans(last_layer, num_selected=num_selected)

    df=pd.read_csv(data_path,sep='\t',header=None)
    df=df.iloc[selected_idx]

    if num_selected == 20000:
        folder_name = "kmeans"
    else:
        n_selected=num_selected//1000
        folder_name = f"kmeans_{n_selected}k"

    out_path = f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}/round_{round}/{folder_name}/{arch}_{seed}"

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    df.to_csv(f'{out_path}/selected.txt',sep='\t', index=False,header=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("round",type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--num_selected",type=int,default=20000)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")
        
    kmeans_al(species=args.species,
             arch=args.arch,
             round=args.round,
             seed=args.seed,
             num_selected=args.num_selected)

if __name__ == "__main__":
    main()