import numpy as np
import pandas as pd
import torch
import argparse
from models.model_loader import load_al_model
from .utils import enable_dropout, write_selections, _forward
from .dataloader import prepare_dataloader
from nextFrag.config import get_project_root, DATASET_CONFIG

def mc_dropout(
    dataset: str,
    arch: str,
    strategy: str, # for compatibility, unused
    round_num: int,
    seed: int,
    num_passes: int=5,
    num_selected: int=20_000,
    batch_size: int=4096
):
    data_path = get_project_root() / dataset / f'round_{round_num}' / 'mcd' / f'{arch}_{seed}' / 'data' / 'pool.txt'
    seqsize = DATASET_CONFIG[dataset]['seqsize']
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = prepare_dataloader(
        data_path, 
        seqsize=seqsize,
        dset=dataset,
        batch_size=batch_size,
        shuffle = False
    )
    model=load_al_model(dataset=dataset,al_strategy='mcd',arch=arch,seed=seed,round_num=round_num-1)
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
                model_preds.append(_forward(model, X))
            combined = torch.stack(model_preds).cpu().numpy()
            var = np.var(combined,axis=0)
            all_var.append(var)

    all_var=np.concatenate(all_var)
    all_var=all_var.reshape(2,num_seqs)
    all_var=np.max(all_var,axis=0) # take max variance between sequence and reverse complement

    df['var']=all_var
    df=df.sort_values(by=['var'],ascending=False).head(num_selected)
    write_selections(
        df, 
        dataset=dataset,
        strategy='mcd',
        round_num=round_num,
        num_selected=num_selected,
        arch=arch,
        seed=seed
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("round",type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--num_passes",type=int,default=5)
    parser.add_argument("--num_selected",type=int,default=20_000)
    parser.add_argument("--batch_size",type=int,default=4096)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    mc_dropout(
        dataset=args.dataset,
        arch=args.arch,
        round_num=args.round,
        seed=args.seed,
        num_passes=args.num_passes,
        num_selected=args.num_selected,
        batch_size=args.batch_size
    )
    print("\nSelection complete!")

if __name__ == "__main__":
    main()