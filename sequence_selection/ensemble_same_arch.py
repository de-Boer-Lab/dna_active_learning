import numpy as np
import pandas as pd
import torch, os, argparse, copy
from models.model_utils import load_model
from models.dl_utils import prepare_dataloader
from typing import List

def ensemble_same_arch(species: str,
                       arch: str,
                       round: int,
                       seeds: List[int],
                       num_selected: int=20_000):
    DATA_ROOT="" # replace with the root directory of your AL run
    data_path = f"/{DATA_ROOT}/{species}/round_{round-1}/same_arch/{arch}_{seeds[0]}/pool.txt"
    seqsize = 200 if species == 'human' else 150
    batch_size = 4096
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = prepare_dataloader(data_path, 
                                seqsize=seqsize,
                                species=species,
                                batch_size=batch_size,
                                shuffle = False)

    models=[]
    for seed in seeds:
        model=load_model(species=species, al_method='same_arch', arch=arch, seed=seed, round=round-1)
        models.append(copy.deepcopy(model))  

    for model in models:
        model = model.to(device)
        model = model.eval()

    df=pd.read_csv(data_path,header=None,sep='\t')
    num_seqs=len(df)

    all_var=[]
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)

            model_preds=[]
            for model in models:
                model_preds.append(model(X))

            combined = torch.stack(model_preds).cpu().numpy()
            var = np.var(combined,axis=0)
            all_var.append(var)
            
    all_var=np.concatenate(all_var)
    all_var=all_var.reshape(2,num_seqs)
    all_var=np.max(all_var,axis=0) # take max variance between sequence and reverse complement

    sorted_var=np.sort(all_var)
    threshold=sorted_var[-num_selected]
    selected_idx=np.where(all_var>=threshold)[0]
    new_df=df.iloc[selected_idx].copy()

    if num_selected == 20_000:
        folder_name = 'same_arch'
    else:
        n_selected=num_selected//1000
        folder_name = f"same_arch_{n_selected}k"

    out_path = f"/{DATA_ROOT}/{species}/round_{round}/{folder_name}/{arch}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file = f'{out_path}/selected.txt'
    for seed in range(1,6):
        os.symlink(out_path, f'{out_path}_{seed}')

    new_df.to_csv(file,sep='\t',header=None,index=None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("round",type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5])
    parser.add_argument("--num_selected",type=int,default=20_000)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    ensemble_same_arch(species=args.species,
                       arch=args.arch,
                       round=args.round,
                       seeds=args.seeds,
                       num_selected=args.num_selected)

if __name__ == "__main__":
    main()