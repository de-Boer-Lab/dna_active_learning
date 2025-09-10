import torch, csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict
from .dl_utils import prepare_dataloader
from .model_utils import load_model

def load_ground_truth(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
    expressions = [float(line[1]) for line in lines]
    return np.array(expressions)

def average_fwd_rev_pred(data: np.array) -> np.array:
    num_samples=len(data)//2
    return (data[:num_samples]+data[num_samples:])/2

def load_promoter_class_indices(file_path):
    df = pd.read_csv(file_path)
    if 'pos' in df.columns:
        return np.unique(np.array(df['pos']))
    elif 'alt_pos' in df.columns and 'ref_pos' in df.columns:
        SNVs_alt = list(df['alt_pos'])
        SNVs_ref = list(df['ref_pos'])
        return list(set(list(zip(SNVs_alt, SNVs_ref))))

def calculate_correlations(index_list, expressions, GROUND_TRUTH_EXP):
    PRED_DATA = OrderedDict()
    GROUND_TRUTH = OrderedDict()

    for j in index_list:
        PRED_DATA[str(j)] = float(expressions[j])
        GROUND_TRUTH[str(j)] = float(GROUND_TRUTH_EXP[j])

    pearson = pearsonr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]
    spearman = spearmanr(list(GROUND_TRUTH.values()), list(PRED_DATA.values()))[0]

    return pearson, spearman


def calculate_diff_correlations(pair_list, expressions, GROUND_TRUTH_EXP):
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

def evaluate_yeast_predictions(expressions,result_file: str):
    expressions = np.array(expressions)
    GROUND_TRUTH_EXP = load_ground_truth('data/yeast/test.txt')
    # Load indices for different promoter classes
    high = load_promoter_class_indices('data/yeast/test_subset_ids/high_exp_seqs.csv')
    low = load_promoter_class_indices('data/yeast/test_subset_ids/low_exp_seqs.csv')
    yeast = load_promoter_class_indices('data/yeast/test_subset_ids/yeast_seqs.csv')
    random = load_promoter_class_indices('data/yeast/test_subset_ids/all_random_seqs.csv')
    challenging = load_promoter_class_indices('data/yeast/test_subset_ids/challenging_seqs.csv')
    SNVs = load_promoter_class_indices('data/yeast/test_subset_ids/all_SNVs_seqs.csv')
    motif_perturbation = load_promoter_class_indices('data/yeast/test_subset_ids/motif_perturbation_seqs.csv')
    motif_tiling = load_promoter_class_indices('data/yeast/test_subset_ids/motif_tiling_seqs.csv')

    final_all = list(range(len(GROUND_TRUTH_EXP)))

    # Calculate correlations
    pearson, spearman = calculate_correlations(final_all, expressions, GROUND_TRUTH_EXP)
    high_pearson, high_spearman = calculate_correlations(high, expressions, GROUND_TRUTH_EXP)
    low_pearson, low_spearman = calculate_correlations(low, expressions, GROUND_TRUTH_EXP)
    yeast_pearson, yeast_spearman = calculate_correlations(yeast, expressions, GROUND_TRUTH_EXP)
    random_pearson, random_spearman = calculate_correlations(random, expressions, GROUND_TRUTH_EXP)
    challenging_pearson, challenging_spearman = calculate_correlations(challenging, expressions, GROUND_TRUTH_EXP)

    # Calculate difference correlations
    SNVs_pearson, SNVs_spearman = calculate_diff_correlations(SNVs, expressions, GROUND_TRUTH_EXP)
    motif_perturbation_pearson, motif_perturbation_spearman = calculate_diff_correlations(motif_perturbation, expressions, GROUND_TRUTH_EXP)
    motif_tiling_pearson, motif_tiling_spearman = calculate_diff_correlations(motif_tiling, expressions, GROUND_TRUTH_EXP)

    # Calculate scores
    pearsons_score = (pearson**2 + 0.3 * high_pearson**2 + 0.3 * low_pearson**2 + 0.3 * yeast_pearson**2 + 
                    0.3 * random_pearson**2 + 0.5 * challenging_pearson**2 + 1.25 * SNVs_pearson**2 + 
                    0.3 * motif_perturbation_pearson**2 + 0.4 * motif_tiling_pearson**2) / 4.65


    spearmans_score = (spearman + 0.3 * high_spearman + 0.3 * low_spearman + 0.3 * yeast_spearman 
                    + 0.3 * random_spearman + 0.5 * challenging_spearman + 1.25 * SNVs_spearman
                    + 0.3 * motif_perturbation_spearman + 0.4 * motif_tiling_spearman) / 4.65

    # Print scores
    with open(result_file, 'w') as f:
        f.write('Pearson Score\t' + str(pearsons_score) + '\n')
        f.write('all r\t' + str(pearson) + '\n')
        f.write('high r\t' + str(high_pearson) + '\n')
        f.write('low r\t' + str(low_pearson) + '\n')
        f.write('yeast r\t' + str(yeast_pearson) + '\n')
        f.write('random r\t' + str(random_pearson) + '\n')
        f.write('challenging r\t' + str(challenging_pearson) + '\n')
        f.write('SNVs r\t' + str(SNVs_pearson) + '\n')
        f.write('motif perturbation r\t' + str(motif_perturbation_pearson) + '\n')
        f.write('motif tiling r\t' + str(motif_tiling_pearson) + '\n')

def eval_human_model(arch: str, 
                     model_path: str | Path = None, 
                     out_file: str | Path = None, 
                     al_method: str = None, 
                     round: int = None, 
                     seed: int = None, 
                     batch_size: int=4096):
    test_path_ID = f"data/human/demo_test.txt" # replace with actual path
    test_path_OOD = f"data/human/demo_test.txt"  # replace with actual path
    test_path_SNV = f"data/human/demo_test_snv.txt" # replace with actual path
    seqsize = 200
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ID_dl = prepare_dataloader(test_path_ID, 
                                seqsize=seqsize, 
                                species='human',
                                batch_size=batch_size,
                                shuffle=False)
    test_OOD_dl = prepare_dataloader(test_path_OOD, 
                                seqsize=seqsize, 
                                species='human',
                                batch_size=batch_size,
                                shuffle=False)
    test_SNV_dl = prepare_dataloader(test_path_SNV, 
                                seqsize=seqsize, 
                                species='human',
                                batch_size=batch_size,
                                shuffle=False)
    if model_path is not None:
        model=load_model(path=model_path,species='human',arch=arch)
    else: # model path inferred from arch, al_method, etc
        model=load_model(species='human',
                        arch=arch,
                        al_method=al_method,
                        seed=seed,
                        round=round)
    model.to(device).eval()

    all_model_predictions=[]
    final_result=[]
    for dl in [test_ID_dl, test_OOD_dl, test_SNV_dl]:
        with torch.inference_mode():
            predictions=[]
            for batch in dl:
                X = batch["x"].to(device)
                predictions.append(model(X).cpu().numpy())
        all_preds=np.concatenate(predictions,axis=0)
        all_preds=np.squeeze(all_preds)
        all_preds = average_fwd_rev_pred(data=all_preds)
        all_model_predictions.append(all_preds)
    
    for i, gt_path in enumerate([test_path_ID, test_path_OOD]):
        gt=load_ground_truth(gt_path)
        final_result.append(pearsonr(all_model_predictions[i],gt)[0])
    
    snv_df = pd.read_csv(test_path_SNV,sep='\t',header=None)
    snv_length = len(snv_df)//2

    snv_expr = np.array(snv_df[1]).reshape((snv_length,2))
    preds=all_model_predictions[2].reshape((snv_length,2))
    snv_gt=snv_expr[:,1]-snv_expr[:,0]
    snv_pred = preds[:,1]-preds[:,0]
    final_result.append(pearsonr(snv_pred,snv_gt)[0])

    if out_file is not None:
        result_file = out_file
    else:
        result_file = "data/human/results.txt" # replace with actual path 
    with open(result_file, 'w') as f:
        f.write(f"ID\t{final_result[0]}\n")
        f.write(f"OOD\t{final_result[1]}\n")
        f.write(f"SNV\t{final_result[2]}\n")

def eval_yeast_model(arch: str,
                     model_path: str | Path = None, 
                     out_file: str | Path = None, 
                     al_method: str = None, 
                     seed: int = None, 
                     round: int = None, 
                     batch_size: int=4096):
    test_path = f"data/yeast/test.txt" 
    seqsize = 150
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dl = prepare_dataloader(test_path, 
                                seqsize=seqsize, 
                                species='yeast',
                                batch_size=batch_size,
                                shuffle=False)
    
    if model_path is not None:
        model=load_model(path=model_path,species='yeast',arch=arch)
    else: # model path inferred from arch, al_method, etc
        model=load_model(species='yeast',
                        arch=arch,
                        al_method=al_method,
                        seed=seed,
                        round=round)
    model.to(device).eval()

    with torch.inference_mode():
        predictions=[]
        for batch in test_dl:
            X = batch["x"].to(device)
            predictions.append(model(X).cpu().numpy())
    all_preds=np.concatenate(predictions,axis=0)
    all_preds=np.squeeze(all_preds)
    result = average_fwd_rev_pred(data=all_preds)

    if out_file is not None:
        result_file = out_file
    else:
        result_file = "data/human/results.txt" # replace with actual path 
    evaluate_yeast_predictions(result,result_file=result_file)

def eval_model(species: str, 
               arch: str, 
               model_path: str | Path = None,
               out_file: str | Path = None,
               al_method: str = None, 
               round: int = None, 
               seed: int = None, 
               batch_size: int=4096):
    '''
    expects either model_path or (al_method AND round AND seed)
    '''
    match species:
        case 'human':
            eval = eval_human_model
        case 'yeast':
            eval = eval_yeast_model

    if model_path is not None:
        return eval(arch=arch,model_path=model_path,out_file=out_file)
    else: # infer model path from other params
        return eval(arch=arch,al_method=al_method,round=round,seed=seed)