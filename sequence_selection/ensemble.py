import numpy as np
import pandas as pd
import torch, argparse, copy
from typing import List
from pathlib import Path
from .utils import write_selections, _forward
from models.model_loader import load_al_model
from .dataloader import prepare_dataloader
from nextFrag.config import get_project_root, DATASET_CONFIG

def ensemble_select(
    models: List[torch.nn.Module],
    data_path: Path,
    dataset: str,
    seqsize: int,
    num_selected: int = 20_000,
    batch_size: int = 2048,
    dream_model: bool = False
) -> pd.DataFrame:
    """Score a pool of sequences by prediction disagreement across a list of
    PyTorch models and return the top-*num_selected* rows.

    Parameters
    ----------
    models: Two or more models.
    data_path: Tab-separated pool file.
    num_selected: How many sequences to keep.
    batch_size: Dataloader batch size.
    seqsize: Sequence length expected by the models.
    dataset: Dataset identifier forwarded to ``prepare_dataloader``.
    dream_model: If True, call ``model.predict(X)``; otherwise ``model(X)``.

    Returns
    -------
    pd.DataFrame
        Top-*num_selected* rows with an added ``'var'`` column, sorted by
        descending disagreement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(models) < 2:
        raise ValueError("Ensemble selection requires at least 2 models.")

    models = [m.to(device).eval() for m in models]
    dataloader = prepare_dataloader(
        data_path,
        seqsize=seqsize,
        dset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    df = pd.read_csv(data_path, header=None, sep="\t")
    num_seqs = len(df)

    all_var=[]
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            predictions = [_forward(m, X, dream_model) for m in models]

            if len(predictions) > 2:
                combined = torch.stack(predictions, dim=0)
                var = torch.var(combined, dim=0).cpu().numpy()
            else:
                var = torch.abs(predictions[0] - predictions[1]).cpu().numpy()

            all_var.append(var)

    all_var_arr = np.concatenate(all_var)
    all_var_arr = all_var_arr.reshape(2, num_seqs)
    all_var_arr = np.max(all_var_arr, axis=0)

    df["var"] = all_var_arr
    result_df = df.sort_values("var", ascending=False).head(num_selected)
    return result_df

def ensemble_multi_arch(
    dataset: str,
    composition: str,
    round_num: int,
    seed: int,
    num_selected: int=20_000,
    batch_size: int=2048
):
    data_path = get_project_root() / dataset / f'round_{round_num}' / composition / f'rnn_{seed}' / 'data' / 'pool.txt'
    seqsize = DATASET_CONFIG[dataset]['seqsize']

    models = []    
    if composition != 'cnn-attn':
        rnn = load_al_model(
            dataset=dataset,
            al_strategy=composition,
            arch='rnn',
            seed=seed,
            round_num=round_num - 1
        )
        models.append(rnn)
    
    if composition != 'rnn-attn':
        cnn = load_al_model(
            dataset=dataset,
            al_strategy=composition,
            arch='cnn',
            seed=seed,
            round_num=round_num - 1
        )
        models.append(cnn)
    
    if composition != 'rnn-cnn':
        attn = load_al_model(
            dataset=dataset,
            al_strategy=composition,
            arch='attn',
            seed=seed,
            round_num=round_num - 1
        )
        models.append(attn)
    
    result_df = ensemble_select(
        models=models,
        data_path=data_path,
        dataset=dataset,
        seqsize=seqsize,
        num_selected=num_selected,
        batch_size=batch_size,
        dream_model=True
    )

    symlinks=[]
    for arch in ['cnn','rnn']:
        symlinks.append(f"{arch}_{seed}")

    write_selections(
        result_df, 
        dataset=dataset,
        strategy=composition,
        round_num=round_num,
        num_selected=num_selected,
        arch='rnn',
        seed=seed,
        symlinks=symlinks
    )

def ensemble_same_arch(
    dataset: str,
    arch: str,
    round_num: int,
    seeds: List[int],
    num_selected: int=20_000,
    batch_size = 4096
):
    data_path = get_project_root() / dataset / f'round_{round_num}' / 'same_arch' / f'{arch}_{seeds[0]}' / 'data' / 'pool.txt'
    seqsize = DATASET_CONFIG[dataset]['seqsize']

    models=[]
    for seed in seeds:
        model=load_al_model(dataset=dataset, al_strategy='same_arch', arch=arch, seed=seed, round_num=round_num-1)
        models.append(copy.deepcopy(model))  

    result_df = ensemble_select(
        models=models,
        data_path=data_path,
        dataset=dataset,
        seqsize=seqsize,
        num_selected=num_selected,
        batch_size=batch_size,
        dream_model=True
    )
    
    symlinks=[]
    for seed in seeds[1:]:
        symlinks.append(f"{arch}_{seed}")

    write_selections(
        result_df, 
        dataset=dataset,
        strategy='same_arch',
        round_num=round_num,
        num_selected=num_selected,
        arch=arch,
        seed=seeds[0],
        symlinks=symlinks
    ) 

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # ===== Multi-Architecture Ensemble =====
    multi_parser = subparsers.add_parser('multi',help='Ensemble across different architectures')
    multi_parser.add_argument('dataset',choices=['yeast', 'human'])
    multi_parser.add_argument('composition',choices=['all_arch', 'rnn-cnn', 'rnn-attn', 'cnn-attn'])
    multi_parser.add_argument('--round',type=int,required=True)
    multi_parser.add_argument('--seed',type=int,required=True)
    multi_parser.add_argument('--num-selected',type=int,default=20_000)
    multi_parser.add_argument('--batch-size',type=int,default=2048)
    
    # ===== Same Architecture Ensemble =====
    same_parser = subparsers.add_parser('same',help='Ensemble across same architecture with different seeds')
    same_parser.add_argument('dataset',choices=['yeast', 'human'])
    same_parser.add_argument('arch',choices=['cnn', 'rnn', 'attn'])
    same_parser.add_argument('--round',type=int,required=True)
    same_parser.add_argument('--seeds',type=int,nargs='+',required=True)
    same_parser.add_argument('--num-selected',type=int,default=20_000)
    same_parser.add_argument('--batch-size',type=int,default=4096)

    args = parser.parse_args()
    
    if args.mode == 'multi':
        print(f"Multi-architecture ensemble: {args.composition}")
        print(f"  Dataset: {args.dataset}, Round: {args.round}, Seed: {args.seed}")
        
        ensemble_multi_arch(
            dataset=args.dataset,
            composition=args.composition,
            round_num=args.round,
            seed=args.seed,
            num_selected=args.num_selected,
            batch_size=args.batch_size
        )
    else:  # same
        print(f"Same-architecture ensemble: {args.arch}")
        print(f"  Dataset: {args.dataset}, Round: {args.round}")
        print(f"  Seeds: {args.seeds}")
        
        ensemble_same_arch(
            dataset=args.dataset,
            arch=args.arch,
            round_num=args.round,
            seeds=args.seeds,
            num_selected=args.num_selected,
            batch_size=args.batch_size
        )
    print("\nSelection complete!")

if __name__ == "__main__":
    main()