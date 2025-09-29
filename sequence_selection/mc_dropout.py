import numpy as np
import pandas as pd
import torch, os, argparse
from models.model_utils import load_model
from .utils import enable_dropout
from models.dl_utils import prepare_dataloader

def mc_dropout(species: str,
               arch: str,
               round: int,
               seed: int,
               num_passes: int=5,
               num_selected: int=20_000):
    DATA_ROOT="" # replace with the root directory of your AL run
    data_path = f"/{DATA_ROOT}/{species}/round_{round-1}/mcd/{arch}_{seed}/pool.txt"
    seqsize = 200 if species == 'human' else 150
    batch_size = 4096
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = prepare_dataloader(data_path, 
                                   seqsize=seqsize,
                                   species=species,
                                   batch_size=batch_size,
                                   shuffle = False)
    model=load_model(species=species,al_method='mcd',arch=arch,seed=seed,round=round-1)
    model = model.to(device).eval()
    enable_dropout(model)

    df=pd.read_csv(data_path,header=None,sep='\t')
    num_seqs=len(df)

    all_var=[]
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            model_preds=[]

            for _ in range(num_passes):
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
        folder_name = 'mcd'
    else:
        n_selected=num_selected//1000
        folder_name = f"mcd_{n_selected}k"

    out_path = f"/{DATA_ROOT}/{species}/round_{round}/{folder_name}/{arch}_{seed}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file = f'{out_path}/selected.txt'
    new_df.to_csv(file,sep='\t',header=None,index=None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("round",type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--num_passes",type=int,default=5)
    parser.add_argument("--num_selected",type=int,default=20_000)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    mc_dropout(species=args.species,
               arch=args.arch,
               round=args.round,
               seed=args.seed,
               num_passes=args.num_passes,
               num_selected=args.num_selected)

if __name__ == "__main__":
    main()