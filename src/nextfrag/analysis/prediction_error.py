import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path
from nextfrag.utils import _forward
from nextfrag.models.model_loader import load_model
from nextfrag.data.dataloader import build_dataloader
from nextfrag.config import DATASET_CONFIG, DEFAULT_N_SELECTED


def prediction_error(
    data_path: str | Path,
    model_path: str | Path,
    out_path: str | Path,
    dataset: str,
    arch: str,
    batch_size: int = 2048,
) -> None:
    """Calculate model prediction error on a sequence file.

    Forward and reverse-complement predictions are averaged. Output is a
    tab-separated file with the original columns plus error and prediction columns.
    """
    seqsize = DATASET_CONFIG[dataset]['seqsize']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(path=model_path, dataset=dataset, arch=arch)
    model = model.to(device).eval()

    dataloader = build_dataloader(
        data_path, seqsize=seqsize, dataset=dataset, batch_size=batch_size, shuffle=False,
    )
    df = pd.read_csv(data_path, header=None, sep="\t")
    df.columns = ['seq', 'expr']
    num_seqs = len(df)

    all_preds = []
    with torch.inference_mode():
        for batch in dataloader:
            X = batch["x"].to(device)
            all_preds.append(_forward(model, X).cpu().numpy())

    all_preds_arr = np.concatenate(all_preds).reshape(2, num_seqs)
    all_preds_arr = np.mean(all_preds_arr, axis=0)

    df["pred"] = all_preds_arr
    df["error"] = df["expr"] - df["pred"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--dataset', choices=['yeast', 'human'], required=True)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--model-path', type=str, default=None,
                        help='Explicit path to model checkpoint. '
                             'If omitted, --acquisition/--round/--seed are required.')
    parser.add_argument('--acquisition', type=str, default=None)
    parser.add_argument('--round', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-selected', type=int, default=DEFAULT_N_SELECTED)

    args = parser.parse_args()

    if args.model_path is not None:
        model_path = args.model_path
    else:
        missing = [name for name, val in [
            ('--acquisition', args.acquisition),
            ('--round', args.round),
            ('--seed', args.seed),
        ] if val is None]
        if missing:
            parser.error(
                f'--model-path not given; also missing: {", ".join(missing)}'
            )
        from nextfrag.path_resolver import PathResolver
        cfg = PathResolver(
            dataset=args.dataset,
            round_num=args.round,
            arch=args.arch,
            seed=args.seed,
            acquisition=args.acquisition,
            n_selected=args.n_selected,
        )
        model_path = cfg.model_path

    prediction_error(
        data_path=args.data_path,
        model_path=model_path,
        out_path=args.out_path,
        dataset=args.dataset,
        arch=args.arch,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
