import numpy as np
import pandas as pd
import torch
import argparse
from nextfrag.models.model_loader import load_model
from nextfrag.utils import write_selections, _forward
from nextfrag.data.dataloader import build_dataloader
from nextfrag.config import DATASET_CONFIG, DEFAULT_N_SELECTED
from nextfrag.path_resolver import PathResolver
from .utils import enable_dropout


def mc_dropout(
    cfg: PathResolver,
    num_passes: int = 5,
    batch_size: int = 4096,
):
    """Select sequences by MC Dropout.

    Loads model with dropout layers active at inference-time, runs num_passes
    forward passes over the pool, and selects the n_selected sequences with
    the highest prediction variance.

    Args:
        cfg: PathResolver for the current round.
        num_passes: Number of stochastic forward passes per sequence.
        batch_size: Inference batch size.
    """
    prev = cfg.prev_round()
    seqsize = DATASET_CONFIG[cfg.dataset]['seqsize']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = build_dataloader(
        prev.pool_path,
        seqsize=seqsize,
        dataset=cfg.dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    model = load_model(path=prev.model_path, dataset=cfg.dataset, arch=cfg.arch)
    model = model.to(device).eval()
    enable_dropout(model)

    df = pd.read_csv(prev.pool_path, header=None, sep='\t')
    num_seqs = len(df)

    all_var = []
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            model_preds = [_forward(model, X) for _ in range(num_passes)]
            combined = torch.stack(model_preds).cpu().numpy()
            all_var.append(np.var(combined, axis=0))

    all_var = np.concatenate(all_var)
    all_var = all_var.reshape(2, num_seqs)
    all_var = np.max(all_var, axis=0)

    df['var'] = all_var
    df = df.sort_values(by=['var'], ascending=False).head(cfg.n_selected)
    write_selections(cfg, result_df=df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['yeast', 'human'], required=True)
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-passes", type=int, default=5)
    parser.add_argument("--n-selected", type=int, default=DEFAULT_N_SELECTED)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")

    cfg = PathResolver(
        dataset=args.dataset,
        round_num=args.round,
        arch=args.arch,
        seed=args.seed,
        acquisition='mcd',
        n_selected=args.n_selected,
    )
    mc_dropout(cfg, num_passes=args.num_passes, batch_size=args.batch_size)
    print("\nSelection complete!")

if __name__ == "__main__":
    main()
    