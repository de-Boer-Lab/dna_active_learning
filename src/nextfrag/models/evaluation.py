import torch
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict
from .model_loader import load_model
from nextfrag.data.dataloader import build_dataloader
from nextfrag.utils import _forward
from nextfrag.config import get_project_root, DATASET_CONFIG, DEFAULT_N_SELECTED

if TYPE_CHECKING:
    from nextfrag.path_resolver import PathResolver

_BUNDLED_DATA_DIR = Path(__file__).parents[3] / 'data'

# TSV column order per dataset — must match the header written by setup_dataset.sh
RESULTS_COLUMNS = {
    'human': ["arch", "acquisition", "seed", "n_selected", "round",
              "id_r", "vep_r", "ood_r"],
    'yeast': ["arch", "acquisition", "seed", "n_selected", "round",
              "pearson_score", "all_r", "high_r", "low_r", "yeast_r",
              "random_r", "challenging_r", "snvs_r",
              "motif_perturbation_r", "motif_tiling_r"],
}


def append_results_tsv(cfg, results: dict) -> None:
    """Append one row to the dataset-level results.tsv.

    cfg must be a PathResolver. results is the dict returned by eval_model.
    """
    columns = RESULTS_COLUMNS[cfg.dataset]
    metadata = {
        "arch": cfg.arch,
        "acquisition": cfg.acquisition or "round_0",
        "seed": cfg.seed,
        "n_selected": cfg.n_selected,
        "round": cfg.round_num,
    }
    row = {**metadata, **results}
    with open(cfg.results_path, "a") as f:
        f.write("\t".join(str(row[k]) for k in columns) + "\n")


def eval_al_model(
    cfg: "PathResolver",
    out_file: str | Path = None,
    batch_size: int = 2048,
    model_class: type = None,
    model_kwargs: dict = None,
) -> dict:
    """Evaluate a trained model on the held-out test set.

    Results are appended to the dataset-level results.tsv automatically.
    If out_file is provided, also writes a human-readable summary there.
    """
    match cfg.dataset:
        case 'human':
            results = eval_human_model(
                model_path=cfg.model_path, arch=cfg.arch, out_file=out_file,
                batch_size=batch_size, model_class=model_class, model_kwargs=model_kwargs,
            )
        case 'yeast':
            results = eval_yeast_model(
                model_path=cfg.model_path, arch=cfg.arch, out_file=out_file,
                batch_size=batch_size, model_class=model_class, model_kwargs=model_kwargs,
            )
        case _:
            raise ValueError(f"Unknown dataset '{cfg.dataset}'")

    append_results_tsv(cfg, results)
    return results

def eval_model(
    dataset: str,
    arch: str,
    model_path: str | Path,
    out_file: str | Path,
    batch_size: int = 2048,
    model_class: type = None,
    model_kwargs: dict = None,
) -> dict:
    """Evaluate a trained model on the held-out test set, outside the AL filestructure."""
    match dataset:
        case 'human':
            results = eval_human_model(
                model_path=model_path, arch=arch, out_file=out_file, batch_size=batch_size,
                model_class=model_class, model_kwargs=model_kwargs,
            )
        case 'yeast':
            results = eval_yeast_model(
                model_path=model_path, arch=arch, out_file=out_file, batch_size=batch_size,
                model_class=model_class, model_kwargs=model_kwargs,
            )
        case _:
            raise ValueError(f"Unknown dataset '{dataset}'")
    
    return results


def eval_human_model(
    model_path: str | Path,
    arch: str,
    out_file: str | Path = None,
    batch_size: int = 2048,
    model_class: type = None,
    model_kwargs: dict = None,
) -> dict:
    """Evaluate a human model on ID, OOD, and VEP test sets.

    Expects test.txt, test_ood.txt, and test_vep.txt under the project root's
    human/ directory. Returns Pearson r for each split.
    """
    test_path_ID  = get_project_root() / 'human' / 'test.txt'
    test_path_OOD = get_project_root() / 'human' / 'test_ood.txt'
    test_path_VEP = get_project_root() / 'human' / 'test_vep.txt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(path=model_path, dataset='human', arch=arch,
                       model_class=model_class, model_kwargs=model_kwargs)
    model.to(device).eval()

    def _preds(path):
        dl = build_dataloader(path, seqsize=DATASET_CONFIG['human']['seqsize'],
                              dataset='human', batch_size=batch_size, shuffle=False)
        return _run_predictions(model, dl, device)

    id_preds  = _preds(test_path_ID)
    ood_preds = _preds(test_path_OOD)
    vep_preds = _preds(test_path_VEP)

    id_r  = pearsonr(id_preds,  _load_ground_truth(test_path_ID))[0]
    ood_r = pearsonr(ood_preds, _load_ground_truth(test_path_OOD))[0]

    vep_df = pd.read_csv(test_path_VEP, sep='\t', header=None)
    n_vep = len(vep_df) // 2
    vep_gt   = np.array(vep_df[1]).reshape((n_vep, 2))
    vep_pred = vep_preds.reshape((n_vep, 2))
    vep_r = pearsonr(vep_pred[:, 1] - vep_pred[:, 0], vep_gt[:, 1] - vep_gt[:, 0])[0]

    results = {"id_r": id_r, "vep_r": vep_r, "ood_r": ood_r}

    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            f.write(f"ID\t{id_r}\nVEP\t{vep_r}\nOOD\t{ood_r}\n")

    return results


def eval_yeast_model(
    model_path: str | Path,
    arch: str,
    out_file: str | Path = None,
    batch_size: int = 2048,
    model_class: type = None,
    model_kwargs: dict = None,
) -> dict:
    """Evaluate a yeast model using the DREAM Challenge weighted scoring scheme.

    Expects test.txt and test_subset_ids/ under the package's data/yeast/ directory.
    Returns pearson_score (weighted composite) and per-subset Pearson r values.
    See _evaluate_yeast_predictions for the scoring formula.
    """
    test_path = get_project_root() / 'yeast' / 'test.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(path=model_path, dataset='yeast', arch=arch,
                       model_class=model_class, model_kwargs=model_kwargs)
    model.to(device).eval()

    test_dl = build_dataloader(test_path, seqsize=DATASET_CONFIG['yeast']['seqsize'],
                               dataset='yeast', batch_size=batch_size, shuffle=False)
    preds = _run_predictions(model, test_dl, device)
    return _evaluate_yeast_predictions(preds, out_file=out_file)


def _run_predictions(model, dataloader, device) -> np.ndarray:
    with torch.inference_mode():
        preds = [_forward(model, batch["x"].to(device)).cpu().numpy()
                 for batch in dataloader]
    return _average_fwd_rev_pred(np.squeeze(np.concatenate(preds)))


def _load_ground_truth(filename: str | Path) -> np.ndarray:
    with open(filename) as f:
        return np.array([float(row[1]) for row in csv.reader(f, delimiter="\t")])


def _average_fwd_rev_pred(data: np.ndarray) -> np.ndarray:
    n = len(data) // 2
    return (data[:n] + data[n:]) / 2


def _load_promoter_class_indices(file_path):
    df = pd.read_csv(file_path)
    if 'pos' in df.columns:
        return np.unique(np.array(df['pos']))
    elif 'alt_pos' in df.columns and 'ref_pos' in df.columns:
        SNVs_alt = list(df['alt_pos'])
        SNVs_ref = list(df['ref_pos'])
        return list(set(list(zip(SNVs_alt, SNVs_ref))))


def _calculate_correlations(index_list, expressions, GROUND_TRUTH_EXP):
    PRED_DATA = OrderedDict()
    GROUND_TRUTH = OrderedDict()

    for j in index_list:
        PRED_DATA[str(j)] = float(expressions[j])
        GROUND_TRUTH[str(j)] = float(GROUND_TRUTH_EXP[j])

    pearson = pearsonr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]
    spearman = spearmanr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]

    return pearson, spearman


def _calculate_diff_correlations(pair_list, expressions, GROUND_TRUTH_EXP):
    Y_pred_selected = []
    expressions_selected = []

    for pair in pair_list:
        ref, alt = pair[0], pair[1]
        Y_pred_selected.append(expressions[alt] - expressions[ref])
        expressions_selected.append(GROUND_TRUTH_EXP[alt] - GROUND_TRUTH_EXP[ref])

    Y_pred_selected = np.array(Y_pred_selected)
    expressions_selected = np.array(expressions_selected)

    pearson = pearsonr(expressions_selected, Y_pred_selected)[0]
    spearman = spearmanr(expressions_selected, Y_pred_selected)[0]

    return pearson, spearman


def _evaluate_yeast_predictions(expressions, out_file: str | Path = None) -> dict:
    """Score yeast predictions using the DREAM Challenge weighted Pearson formula.

    Weighted composite score:
        (r² + 0.3·high² + 0.3·low² + 0.3·yeast² + 0.3·random²
         + 0.5·challenging² + 1.25·SNVs² + 0.3·motif_perturbation²
         + 0.4·motif_tiling²) / 4.65

    SNVs and motif subsets use difference correlations (alt − ref).
    """
    expressions = np.array(expressions)
    data_dir = _BUNDLED_DATA_DIR / 'yeast'
    subset_ids_dir = data_dir / 'test_subset_ids'

    GROUND_TRUTH_EXP = _load_ground_truth(get_project_root() / 'yeast' / 'test.txt')
    high = _load_promoter_class_indices(subset_ids_dir / 'high_exp_seqs.csv')
    low = _load_promoter_class_indices(subset_ids_dir / 'low_exp_seqs.csv')
    yeast = _load_promoter_class_indices(subset_ids_dir / 'yeast_seqs.csv')
    random = _load_promoter_class_indices(subset_ids_dir / 'all_random_seqs.csv')
    challenging = _load_promoter_class_indices(subset_ids_dir / 'challenging_seqs.csv')
    SNVs = _load_promoter_class_indices(subset_ids_dir / 'all_SNVs_seqs.csv')
    motif_perturbation = _load_promoter_class_indices(subset_ids_dir / 'motif_perturbation_seqs.csv')
    motif_tiling = _load_promoter_class_indices(subset_ids_dir / 'motif_tiling_seqs.csv')

    final_all = list(range(len(GROUND_TRUTH_EXP)))

    pearson, spearman = _calculate_correlations(final_all, expressions, GROUND_TRUTH_EXP)
    high_pearson, high_spearman = _calculate_correlations(high, expressions, GROUND_TRUTH_EXP)
    low_pearson, low_spearman = _calculate_correlations(low, expressions, GROUND_TRUTH_EXP)
    yeast_pearson, yeast_spearman = _calculate_correlations(yeast, expressions, GROUND_TRUTH_EXP)
    random_pearson, random_spearman = _calculate_correlations(random, expressions, GROUND_TRUTH_EXP)
    challenging_pearson, challenging_spearman = _calculate_correlations(challenging, expressions, GROUND_TRUTH_EXP)
    SNVs_pearson, SNVs_spearman = _calculate_diff_correlations(SNVs, expressions, GROUND_TRUTH_EXP)
    motif_perturbation_pearson, motif_perturbation_spearman = _calculate_diff_correlations(motif_perturbation, expressions, GROUND_TRUTH_EXP)
    motif_tiling_pearson, motif_tiling_spearman = _calculate_diff_correlations(motif_tiling, expressions, GROUND_TRUTH_EXP)

    pearsons_score = (pearson**2 + 0.3 * high_pearson**2 + 0.3 * low_pearson**2 + 0.3 * yeast_pearson**2 +
                     0.3 * random_pearson**2 + 0.5 * challenging_pearson**2 + 1.25 * SNVs_pearson**2 +
                     0.3 * motif_perturbation_pearson**2 + 0.4 * motif_tiling_pearson**2) / 4.65

    spearmans_score = (spearman + 0.3 * high_spearman + 0.3 * low_spearman + 0.3 * yeast_spearman
                      + 0.3 * random_spearman + 0.5 * challenging_spearman + 1.25 * SNVs_spearman
                      + 0.3 * motif_perturbation_spearman + 0.4 * motif_tiling_spearman) / 4.65

    results = {
        "pearson_score": pearsons_score,
        "all_r": pearson,
        "high_r": high_pearson,
        "low_r": low_pearson,
        "yeast_r": yeast_pearson,
        "random_r": random_pearson,
        "challenging_r": challenging_pearson,
        "snvs_r": SNVs_pearson,
        "motif_perturbation_r": motif_perturbation_pearson,
        "motif_tiling_r": motif_tiling_pearson,
    }

    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            f.write(f'Pearson Score\t{pearsons_score}\n')
            f.write(f'all r\t{pearson}\n')
            f.write(f'high r\t{high_pearson}\n')
            f.write(f'low r\t{low_pearson}\n')
            f.write(f'yeast r\t{yeast_pearson}\n')
            f.write(f'random r\t{random_pearson}\n')
            f.write(f'challenging r\t{challenging_pearson}\n')
            f.write(f'SNVs r\t{SNVs_pearson}\n')
            f.write(f'motif perturbation r\t{motif_perturbation_pearson}\n')
            f.write(f'motif tiling r\t{motif_tiling_pearson}\n')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['yeast', 'human'])
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--acquisition", type=str, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-selected", type=int, default=DEFAULT_N_SELECTED)
    args = parser.parse_args()

    from nextfrag.path_resolver import PathResolver
    cfg = PathResolver(
        dataset=args.dataset,
        round_num=args.round,
        arch=args.arch,
        seed=args.seed,
        acquisition=args.acquisition,
        n_selected=args.n_selected,
    )

    print("Evaluating:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")

    results = eval_al_model(cfg)
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
