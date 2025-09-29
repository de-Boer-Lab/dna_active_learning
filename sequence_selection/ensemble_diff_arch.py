import numpy as np
import pandas as pd
import torch, os, argparse
from models.model_utils import load_model
from models.dl_utils import prepare_dataloader

def ensemble_diff_arch(species: str,
                       composition: str,
                       round: int,
                       seed: int,
                       num_selected: int=20_000):
    DATA_ROOT="" # replace with the root directory of your AL run
    data_path = f"/{DATA_ROOT}/{species}/round_{round-1}/{composition}/seed_{seed}/pool.txt"
    seqsize = 200 if species == 'human' else 150
    batch_size = 2048
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = prepare_dataloader(data_path, 
                                seqsize=seqsize,
                                species=species,
                                batch_size=batch_size,
                                shuffle = False)

    cnn=load_model(species=species,al_method=composition,arch='cnn',seed=seed,round=round-1)
    cnn = cnn.to(device).eval()

    rnn=load_model(species=species,al_method=composition,arch='rnn',seed=seed,round=round-1)
    rnn = rnn.to(device).eval()

    attn=load_model(species=species,al_method=composition,arch='attn',seed=seed,round=round-1)
    attn = attn.to(device).eval()

    df=pd.read_csv(data_path,header=None,sep='\t')
    num_seqs=len(df)

    all_var=[]
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)

            if composition != 'cnn-attn':
                rnn_pred = rnn(X)
            if composition != 'rnn-attn':
                cnn_pred = cnn(X)
            if composition != 'rnn-cnn':
                attn_pred = attn(X)

            if composition == 'all_arch':
                combined=torch.stack((cnn_pred,rnn_pred,attn_pred),dim=0)
                combined=combined.cpu().numpy()
                var = np.var(combined,axis=0)
            elif composition == 'rnn-cnn':
                var = torch.abs(rnn_pred-cnn_pred).cpu().numpy()
            elif composition == 'rnn-attn':
                var = torch.abs(rnn_pred-attn_pred).cpu().numpy()
            else: # cnn-attn
                var = torch.abs(cnn_pred-attn_pred).cpu().numpy()
            all_var.append(var)

    all_var=np.concatenate(all_var)
    all_var=all_var.reshape(2,num_seqs)
    all_var=np.max(all_var,axis=0) # take max variance between sequence and reverse complement

    sorted_var=np.sort(all_var)
    threshold=sorted_var[-num_selected]
    selected_idx=np.where(all_var>=threshold)[0]
    new_df=df.iloc[selected_idx].copy()

    if num_selected == 20_000:
        folder_name = composition
    else:
        n_selected=num_selected//1000
        folder_name = f"{composition}_{n_selected}k"

    general_path = f"/{DATA_ROOT}/{species}/round_{round}/{folder_name}"
    out_path = f"{general_path}/seed_{seed}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file = f'{out_path}/selected.txt'
    for arch in ['cnn','rnn','attn']:
        os.symlink(out_path, f'{general_path}/{arch}_{seed}')

    new_df.to_csv(file,sep='\t',header=None,index=None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("composition",choices=['all_arch','rnn-cnn','rnn-attn','cnn-attn'])
    parser.add_argument("round",type=int)
    parser.add_argument("seed",type=int)
    parser.add_argument("--num_selected",type=int,default=20_000)
    args = parser.parse_args()
    
    print("Received:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    ensemble_diff_arch(species=args.species,
                       composition=args.composition,
                       round=args.round,
                       seed=args.seed,
                       num_selected=args.num_selected)

if __name__ == "__main__":
    main()