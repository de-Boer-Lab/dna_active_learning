import torch
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict
from models.model_loader import load_model, load_al_model
from .dataloader import prepare_dataloader
from .utils import _forward
from nextFrag.config import get_project_root, DATASET_CONFIG

MODULE_DIR = Path(__file__).parent

def eval_model(
    dataset: str,
    arch: str,
    model_path: str | Path = None,
    out_file: str | Path = None,
    al_strategy: str = None,
    round_num: int = None,
    seed: int = None,
    batch_size: int = 2048,
    model_class: type = None,
    model_kwargs: dict = None,
):
    """Evaluate a trained model on the test set.

    Supply either ``model_path`` or the AL experiment identifiers
    (``al_strategy``, ``round_num``, ``seed``).  For custom models pass
    ``model_class`` and ``model_kwargs`` so the checkpoint can be loaded.
    """
    match dataset:
        case 'human':
            eval_fn = eval_human_model
        case 'yeast':
            eval_fn = eval_yeast_model
        case _:
            raise ValueError(f"Unknown dataset '{dataset}'")
    return eval_fn(
        arch=arch, model_path=model_path, out_file=out_file,
        al_strategy=al_strategy, round_num=round_num, seed=seed,
        batch_size=batch_size, model_class=model_class, model_kwargs=model_kwargs,
    )


def eval_human_model(
    arch: str,
    model_path: str | Path = None,
    out_file: str | Path = None,
    al_strategy: str = None,
    round_num: int = None,
    seed: int = None,
    batch_size: int = 2048,
    model_class: type = None,
    model_kwargs: dict = None,
):
    test_path_ID  = "data/human/demo_test.txt"
    test_path_OOD = "data/human/demo_test.txt"
    test_path_SNV = "data/human/demo_test_snv.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_for_eval('human', arch, model_path, al_strategy, round_num, seed,
                           model_class, model_kwargs)
    model.to(device).eval()

    def _preds(path):
        dl = prepare_dataloader(path, seqsize=DATASET_CONFIG['human']['seqsize'],
                                dset='human', batch_size=batch_size, shuffle=False)
        return _run_predictions(model, dl, device)

    id_preds  = _preds(test_path_ID)
    ood_preds = _preds(test_path_OOD)
    snv_preds = _preds(test_path_SNV)

    id_r  = pearsonr(id_preds,  __load_ground_truth(test_path_ID))[0]
    ood_r = pearsonr(ood_preds, __load_ground_truth(test_path_OOD))[0]

    snv_df = pd.read_csv(test_path_SNV, sep='\t', header=None)
    n_snvs = len(snv_df) // 2
    snv_gt   = np.array(snv_df[1]).reshape((n_snvs, 2))
    snv_pred = snv_preds.reshape((n_snvs, 2))
    snv_r = pearsonr(snv_pred[:, 1] - snv_pred[:, 0], snv_gt[:, 1] - snv_gt[:, 0])[0]

    result_file = Path(out_file or (
        get_project_root() / 'human' / f'round_{round_num}' / al_strategy
        / f'{arch}_{seed}' / 'model' / 'results.txt'
    ))
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
        f.write(f"ID\t{id_r}\nOOD\t{ood_r}\nSNV\t{snv_r}\n")

def eval_yeast_model(
    arch: str,
    model_path: str | Path = None,
    out_file: str | Path = None,
    al_strategy: str = None,
    round_num: int = None,
    seed: int = None,
    batch_size: int = 2048,
    model_class: type = None,
    model_kwargs: dict = None,
):
    test_path = "data/yeast/test.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_for_eval('yeast', arch, model_path, al_strategy, round_num, seed,
                           model_class, model_kwargs)
    model.to(device).eval()

    test_dl = prepare_dataloader(test_path, seqsize=DATASET_CONFIG['yeast']['seqsize'],
                                 dset='yeast', batch_size=batch_size, shuffle=False)
    preds = _run_predictions(model, test_dl, device)

    result_file = out_file or (
        get_project_root() / 'yeast' / f'round_{round_num}' / al_strategy
        / f'{arch}_{seed}' / 'model' / 'results.txt'
    )
    _evaluate_yeast_predictions(preds, result_file=result_file)

def _load_for_eval(dataset, arch, model_path, al_strategy, round_num, seed,
                   model_class, model_kwargs):
    if model_path is not None:
        return load_model(path=model_path, dataset=dataset, arch=arch,
                          model_class=model_class, model_kwargs=model_kwargs)
    return load_al_model(dataset=dataset, arch=arch, al_strategy=al_strategy,
                         round_num=round_num, seed=seed,
                         model_class=model_class, model_kwargs=model_kwargs)

def _run_predictions(model, dataloader, device) -> np.ndarray:
    with torch.inference_mode():
        preds = [_forward(model, batch["x"].to(device)).cpu().numpy()
                 for batch in dataloader]
    return _average_fwd_rev_pred(np.squeeze(np.concatenate(preds)))

def __load_ground_truth(filename: str | Path) -> np.ndarray:
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

def _evaluate_yeast_predictions(expressions, result_file: str | Path):
    expressions = np.array(expressions)
    data_dir = MODULE_DIR.parent / 'data' / 'yeast'
    subset_ids_dir = data_dir / 'test_subset_ids'

    GROUND_TRUTH_EXP = __load_ground_truth(data_dir / 'test.txt')
    # Load indices for different promoter classes
    high = _load_promoter_class_indices(subset_ids_dir / 'high_exp_seqs.csv')
    low = _load_promoter_class_indices(subset_ids_dir / 'low_exp_seqs.csv')
    yeast = _load_promoter_class_indices(subset_ids_dir / 'yeast_seqs.csv')
    random = _load_promoter_class_indices(subset_ids_dir / 'all_random_seqs.csv')
    challenging = _load_promoter_class_indices(subset_ids_dir / 'challenging_seqs.csv')
    SNVs = _load_promoter_class_indices(subset_ids_dir / 'all_SNVs_seqs.csv')
    motif_perturbation = _load_promoter_class_indices(subset_ids_dir / 'motif_perturbation_seqs.csv')
    motif_tiling = _load_promoter_class_indices(subset_ids_dir / 'motif_tiling_seqs.csv')

    final_all = list(range(len(GROUND_TRUTH_EXP)))

    # Calculate correlations
    pearson, spearman = _calculate_correlations(final_all, expressions, GROUND_TRUTH_EXP)
    high_pearson, high_spearman = _calculate_correlations(high, expressions, GROUND_TRUTH_EXP)
    low_pearson, low_spearman = _calculate_correlations(low, expressions, GROUND_TRUTH_EXP)
    yeast_pearson, yeast_spearman = _calculate_correlations(yeast, expressions, GROUND_TRUTH_EXP)
    random_pearson, random_spearman = _calculate_correlations(random, expressions, GROUND_TRUTH_EXP)
    challenging_pearson, challenging_spearman = _calculate_correlations(challenging, expressions, GROUND_TRUTH_EXP)

    # Calculate difference correlations
    SNVs_pearson, SNVs_spearman = _calculate_diff_correlations(SNVs, expressions, GROUND_TRUTH_EXP)
    motif_perturbation_pearson, motif_perturbation_spearman = _calculate_diff_correlations(motif_perturbation, expressions, GROUND_TRUTH_EXP)
    motif_tiling_pearson, motif_tiling_spearman = _calculate_diff_correlations(motif_tiling, expressions, GROUND_TRUTH_EXP)

    # Calculate scores
    pearsons_score = (pearson**2 + 0.3 * high_pearson**2 + 0.3 * low_pearson**2 + 0.3 * yeast_pearson**2 +
                    0.3 * random_pearson**2 + 0.5 * challenging_pearson**2 + 1.25 * SNVs_pearson**2 +
                    0.3 * motif_perturbation_pearson**2 + 0.4 * motif_tiling_pearson**2) / 4.65


    spearmans_score = (spearman + 0.3 * high_spearman + 0.3 * low_spearman + 0.3 * yeast_spearman
                    + 0.3 * random_spearman + 0.5 * challenging_spearman + 1.25 * SNVs_spearman
                    + 0.3 * motif_perturbation_spearman + 0.4 * motif_tiling_spearman) / 4.65

    # Write results
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w') as f:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=['yeast', 'human'])
    parser.add_argument("arch", type=str)
    parser.add_argument("--al-strategy", type=str, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()


    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")
    model_dir = get_project_root() / args.dataset / f'round_{args.round}' / args.al_strategy / f'{args.arch}_{args.seed}' / 'model'
    return eval_model(dataset=args.dataset,
                      arch=args.arch,
                      model_path=model_dir / 'model_best.pth',
                      out_file= model_dir / 'results.txt')
   
if __name__ == "__main__":
    main()
