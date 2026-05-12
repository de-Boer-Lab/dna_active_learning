import numpy as np
import pandas as pd
import torch
import argparse
from typing import List
from pathlib import Path
from nextfrag.utils import write_selections, _forward
from nextfrag.models.model_loader import load_model
from nextfrag.data.dataloader import build_dataloader
from nextfrag.config import DATASET_CONFIG, DEFAULT_N_SELECTED
from nextfrag.path_resolver import PathResolver


def ensemble_select(
    models: List[torch.nn.Module],
    data_path: Path,
    dataset: str,
    seqsize: int,
    n_selected: int,
    batch_size: int = 2048,
) -> pd.DataFrame:
    """Score a pool of sequences by prediction disagreement across a list of
    PyTorch models and return the top-n_selected rows.

    Parameters
    ----------
    models: Two or more models.
    data_path: Tab-separated pool file.
    n_selected: How many sequences to keep.
    seqsize: Sequence length expected by the models.
    dataset: Dataset identifier forwarded to build_dataloader.
    batch_size: Dataloader batch size.
    """
    if len(models) < 2:
        raise ValueError("Ensemble selection requires at least 2 models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [m.to(device).eval() for m in models]
    dataloader = build_dataloader(
        data_path, seqsize=seqsize, dataset=dataset, batch_size=batch_size, shuffle=False,
    )
    df = pd.read_csv(data_path, header=None, sep="\t")
    num_seqs = len(df)

    all_var = []
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            predictions = [_forward(m, X) for m in models]
            if len(predictions) > 2:
                var = torch.var(torch.stack(predictions, dim=0), dim=0).cpu().numpy()
            else:
                var = torch.abs(predictions[0] - predictions[1]).cpu().numpy()
            all_var.append(var)

    all_var_arr = np.concatenate(all_var).reshape(2, num_seqs)
    df["var"] = np.max(all_var_arr, axis=0)
    return df.sort_values("var", ascending=False).head(n_selected)


def ensemble(
    cfgs: List[PathResolver],
    batch_size: int = 2048,
):
    """Run ensemble selection over a list of models.

    cfgs: one PathResolver per model. cfgs[0] is canonical — the pool is read
          from its prev_round and selected.txt is written to its directory.
          cfgs[1:] round_roots are symlinked to cfgs[0].round_root.

    Each model is loaded from its own cfg.prev_round().model_path.
    """
    canonical = cfgs[0]
    prev_canonical = canonical.prev_round()
    seqsize = DATASET_CONFIG[canonical.dataset]['seqsize']

    models = [
        load_model(path=cfg.prev_round().model_path, dataset=cfg.dataset, arch=cfg.arch)
        for cfg in cfgs
    ]

    result_df = ensemble_select(
        models=models,
        data_path=prev_canonical.pool_path,
        dataset=canonical.dataset,
        seqsize=seqsize,
        n_selected=canonical.n_selected,
        batch_size=batch_size,
    )
    write_selections(canonical, result_df=result_df, symlinked_cfgs=cfgs[1:])


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble-based active learning selection.',
        epilog=(
            'Specify members with --models ARCH:SEED [...] for arbitrary combinations,\n'
            'or --arch ARCH --seeds SEED [...] as a shorthand for same-arch ensembles.\n\n'
            'Examples:\n'
            '  %(prog)s --dataset human --acquisition all_arch --round 1 --models rnn:1 cnn:2 attn:3\n'
            '  %(prog)s --dataset human --acquisition same_arch --round 1 --arch rnn --seeds 1 2 3'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', choices=['yeast', 'human'], required=True)
    parser.add_argument('--acquisition', type=str, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--n-selected', type=int, default=DEFAULT_N_SELECTED)
    parser.add_argument('--batch-size', type=int, default=2048)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--models', nargs='+', metavar='ARCH:SEED',
        help='Ensemble members as arch:seed pairs, e.g. rnn:1 cnn:2 attn:3',
    )
    group.add_argument(
        '--arch', type=str,
        help='Single architecture; use with --seeds for a same-arch ensemble',
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+',
        help='Seeds to use when --arch is given',
    )

    args = parser.parse_args()

    if args.arch is not None:
        if not args.seeds:
            parser.error('--arch requires --seeds')
        pairs = [(args.arch, seed) for seed in args.seeds]
    else:
        try:
            pairs = [(arch, int(seed)) for arch, seed in (s.split(':') for s in args.models)]
        except ValueError:
            parser.error('--models entries must be ARCH:SEED, e.g. rnn:1')

    cfgs = [
        PathResolver(
            dataset=args.dataset,
            round_num=args.round,
            arch=arch,
            seed=seed,
            acquisition=args.acquisition,
            n_selected=args.n_selected,
        )
        for arch, seed in pairs
    ]

    print(f"Ensemble: {args.acquisition}")
    print(f"  Members: {pairs}")
    print(f"  Round: {args.round}")
    ensemble(cfgs, batch_size=args.batch_size)
    print("\nSelection complete!")


if __name__ == "__main__":
    main()
