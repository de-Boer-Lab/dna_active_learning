import cuml
import argparse
import pandas as pd
from pathlib import Path
from nextfrag.acquisition.diversity import IPCA
from nextfrag.data.dataloader import build_dataloader
from nextfrag.models.model_loader import load_model
from nextfrag.config import DATASET_CONFIG, DEFAULT_N_SELECTED, get_project_root

def umap(
    dataset: str,
    model_path: str | Path,
    out_path: str | Path,
    arch: str,
    data_path: str | Path = None,
    n_pca_components: int = 64,
    batch_size: int = 2048,
    n_umap_components: int = 2,
    n_umap_neighbors: int = 15,
    n_umap_epochs: int = 200,
):
    """Compute a UMAP embedding of pool sequences in model feature space.

    Extracts penultimate-layer features via IPCA, then applies cuML UMAP.
    Defaults to the round-0 pool if data_path is not specified.

    Output is a tab-separated file with columns: seq, expr, umap0[, umap1, ...].
    """
    if data_path is None:
        data_path = get_project_root() / dataset / "round_0" / "data" / "pool.txt"
    seqsize = DATASET_CONFIG[dataset]['seqsize']

    dataloader = build_dataloader(
        data_path,
        seqsize=seqsize,
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        revcomp_same_batch=True,
    )

    model = load_model(dataset=dataset, arch=arch, path=model_path)

    X = IPCA(model=model, dataloader=dataloader, n_components=n_pca_components, batch_size=batch_size // 2)
    reducer = cuml.manifold.UMAP(
        n_components=n_umap_components, n_neighbors=n_umap_neighbors, n_epochs=n_umap_epochs,
    )
    embedding = reducer.fit_transform(X)

    df = pd.read_csv(data_path, sep='\t', header=None)
    df.columns = ['seq', 'expr']
    for i in range(n_umap_components): 
        df[f'umap{i}'] = embedding[:, i] 

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=None, sep='\t')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--dataset', choices=['yeast', 'human'], required=True)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--n-pca-components', type=int, default=64)
    parser.add_argument('--n-umap-components', type=int, default=2)
    parser.add_argument('--n-umap-neighbors', type=int, default=15)
    parser.add_argument('--n-umap-epochs', type=int, default=200)
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

    umap(
        dataset=args.dataset,
        model_path=model_path,
        out_path=args.out_path,
        arch=args.arch,
        data_path=args.data_path,
        n_pca_components=args.n_pca_components,
        batch_size=args.batch_size,
        n_umap_components=args.n_umap_components,
        n_umap_neighbors=args.n_umap_neighbors,
        n_umap_epochs=args.n_umap_epochs,
    )


if __name__ == "__main__":
    main()
