import torch
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import argparse
from tqdm import tqdm
from nextfrag.models.model_loader import load_model
from nextfrag.data.dataloader import build_dataloader
from nextfrag.utils import write_selections, _forward
from nextfrag.config import DATASET_CONFIG, DEFAULT_N_SELECTED
from nextfrag.path_resolver import PathResolver

def max_expression(
    cfg: PathResolver,
    batch_size: int = 2048,
):
    """Select sequences with the highest (or lowest) predicted expression.

    The direction is determined by cfg.acquisition: use 'min_expr' to select
    the lowest-predicted sequences, else select the highest.
    """
    min_expr = cfg.acquisition == 'min_expr'
    prev = cfg.prev_round()
    seqsize = DATASET_CONFIG[cfg.dataset]['seqsize']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = build_dataloader(
        prev.pool_path,
        seqsize=seqsize,
        dataset=cfg.dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    model = load_model(path=prev.model_path, dataset=cfg.dataset, arch=cfg.arch)
    model.to(device).eval()

    df = pd.read_csv(prev.pool_path, header=None, sep='\t')
    num_seqs = len(df)

    all_preds = []
    with torch.inference_mode():
        for batch in dataloader:
            X = batch['x'].to(device)
            all_preds.append(_forward(model, X).cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_preds = all_preds.reshape(2, num_seqs)
    all_preds = np.sum(all_preds, axis=0) / 2

    df['pred'] = all_preds
    df = df.sort_values(by=['pred'], ascending=min_expr).head(cfg.n_selected)
    write_selections(cfg, result_df=df)


def ism(
    file_path: str | Path,
    out_path: str | Path,
    dataset: str,
    job_id: int,
    seqs_per_job: int = 500_000,
    window_sz: int = 6,
    arch: str = 'rnn',
    seed: int = 1,
):
    """Run in-silico saturation mutagenesis (ISM) on a subset of sequences.
    Supports splitting across jobs for parallelism.

    Writes a TSV with columns: seq, mean, max, window_max.
    """
    df = pd.read_csv(file_path, header=None, sep='\t')
    df.columns = ['seq', 'expr']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dataset == 'human':
        batch_size = 6
        start_pos = 0
        end_pos = 200
        seqsize = DATASET_CONFIG['human']['seqsize']
        df = df[df['seq'].str.len() == seqsize]
    else:
        batch_size = 16
        start_pos = 57
        end_pos = 137
        seqsize = DATASET_CONFIG['yeast']['seqsize']
        left_flank = "AGTGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC"
        df = df[df['seq'].str[:len(left_flank)] == left_flank]
        df = df[~df['seq'].str.contains('N')]
    df = df.iloc[job_id * seqs_per_job:(job_id + 1) * seqs_per_job]

    cfg = PathResolver(dataset=dataset, round_num=0, arch=arch, seed=seed)
    model = load_model(path=cfg.model_path, dataset=dataset, arch=arch)
    model.to(device).eval()

    attrs = np.empty((len(df), end_pos - start_pos, 3), dtype=np.float32)
    idx = 0
    buffer = []
    for row in tqdm(df.itertuples()):
        seq = row.seq.upper()
        X = seq2tensor(seq, dataset)
        buffer.append(X)
        if len(buffer) >= batch_size:
            X = torch.cat(buffer, dim=0)
            y, y_ism = saturation_mutagenesis(model=model, X=X,
                                              start=start_pos, end=end_pos, device=device)
            y_attr = y_ism - y[:, None, None]
            n = len(buffer)
            attrs[idx:idx + n] = y_attr.squeeze(-1).cpu().numpy()
            idx += n
            buffer = []
    if buffer:
        X = torch.cat(buffer, dim=0)
        y, y_ism = saturation_mutagenesis(model=model, X=X,
                                          start=start_pos, end=end_pos, device=device)
        y_attr = y_ism - y[:, None, None]
        n = len(buffer)
        attrs[idx:idx + n] = y_attr.squeeze(-1).cpu().numpy()

    result = compute_attributions(attrs, window_sz)
    out_df = pd.concat([df[["seq"]].reset_index(drop=True),
                        pd.DataFrame(result, columns=['mean', 'max', 'window max'])], axis=1)
    out_df.to_csv(out_path, sep='\t', index=None)


def _edit_distance_one(X, start, end):
    if end < 0:
        end = X.shape[-1] + end + 1
    X_ = X.repeat((end - start) * 3, 1, 1)
    coords = itertools.product(range(start, end), range(4))
    _next = 0
    for pos, mut in coords:
        if X[mut, pos] == 1:
            continue
        X_[_next, :4, pos] = 0
        X_[_next, mut, pos] = 1
        _next += 1
    return X_


def saturation_mutagenesis(model, X, start=0, end=-1, device='cuda'):
    N, C, L = X.shape
    if end < 0:
        end = L + end + 1
    X_mut = [_edit_distance_one(X[i], start, end) for i in range(N)]
    X_mut = torch.cat(X_mut, dim=0)
    X_all = torch.cat((X, X_mut), dim=0)
    model = model.to(device).eval()
    dtype = next(model.parameters(), X).dtype
    with torch.inference_mode():
        y = _forward(model, X_all.to(device).type(dtype))
    y0 = y[:N]
    y_hat = y[N:].view(N, end - start, 3, *y.shape[1:])
    return y0, y_hat


def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0],
               'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return [mapping[base] for base in seq]


def seq2tensor(seq, dataset):
    ohe_seq = one_hot_encode(seq)
    rev_values = [0] * len(ohe_seq)
    is_singletons = [0] * len(ohe_seq)
    if dataset == 'yeast':
        encoded = [list(ohe) + [rev] + [is_singleton]
                   for ohe, rev, is_singleton in zip(ohe_seq, rev_values, is_singletons)]
    else:
        encoded = [list(ohe) + [rev] for ohe, rev in zip(ohe_seq, rev_values)]
    X = torch.Tensor(encoded).type(torch.float32)
    X = X.unsqueeze(0)
    X = torch.transpose(X, 2, 1)
    return X


def compute_attributions(attrs, window_sz):
    attrs = -attrs.mean(axis=2)
    abs_attrs = np.abs(attrs)
    N, L = attrs.shape
    _mean = abs_attrs.mean(axis=1)
    maxpos = abs_attrs.argmax(axis=1)
    _max = attrs[np.arange(N), maxpos]
    windows = np.lib.stride_tricks.sliding_window_view(attrs, window_sz, axis=1)
    window_sums = windows.sum(axis=2)
    idx = np.abs(window_sums).argmax(axis=1)
    max_window = window_sums[np.arange(N), idx]
    return np.stack((_mean, _max, max_window), axis=-1)


def main():
    parser = argparse.ArgumentParser(
        description='Expression-based selection and ISM.',
        epilog=(
            'Use --acquisition max_expr or min_expr for expression-based selection.\n'
            'Use --ism for in-silico saturation mutagenesis on an explicit file.\n\n'
            'Examples:\n'
            '  %(prog)s --dataset yeast --arch rnn --round 1 --seed 42 --acquisition max_expr\n'
            '  %(prog)s --ism --file-path pool.txt --out-path ism.tsv --dataset yeast --arch rnn --seed 1'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', choices=['yeast', 'human'], required=True)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--acquisition', choices=['max_expr', 'min_expr'],
                      help='Expression-based AL selection strategy')
    mode.add_argument('--ism', action='store_true',
                      help='Run in-silico saturation mutagenesis on --file-path')

    # AL selection args
    parser.add_argument('--round', type=int)
    parser.add_argument('--n-selected', type=int, default=DEFAULT_N_SELECTED)
    parser.add_argument('--batch-size', type=int, default=2048)

    # ISM args
    parser.add_argument('--file-path', type=str)
    parser.add_argument('--out-path', type=str)
    parser.add_argument('--job-id', type=int, default=0)
    parser.add_argument('--seqs-per-job', type=int, default=500_000)
    parser.add_argument('--window-sz', type=int, default=6)

    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")

    if args.ism:
        if not args.file_path or not args.out_path:
            parser.error('--ism requires --file-path and --out-path')
        ism(
            file_path=args.file_path,
            out_path=args.out_path,
            dataset=args.dataset,
            job_id=args.job_id,
            seqs_per_job=args.seqs_per_job,
            window_sz=args.window_sz,
            arch=args.arch,
            seed=args.seed,
        )
        print("\nISM complete!")
    else:
        if args.round is None:
            parser.error('--acquisition requires --round')
        cfg = PathResolver(
            dataset=args.dataset,
            round_num=args.round,
            arch=args.arch,
            seed=args.seed,
            acquisition=args.acquisition,
            n_selected=args.n_selected,
        )
        max_expression(cfg, batch_size=args.batch_size)
        print("\nSelection complete!")


if __name__ == "__main__":
    main()
