import torch, csv
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict
from .dl_utils import prepare_dataloader
from .model_utils import init_model

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

def evaluate_yeast_predictions(expressions,path: str):
    expressions = np.array(expressions)
    GROUND_TRUTH_EXP = load_ground_truth('/scratch/st-cdeboer-1/justin/data/test.txt')
    # Load indices for different promoter classes
    high = load_promoter_class_indices('/scratch/st-cdeboer-1/justin/data/test_subset_ids/high_exp_seqs.csv')
    low = load_promoter_class_indices('/scratch/st-cdeboer-1/justin/data/test_subset_ids/low_exp_seqs.csv')
    yeast = load_promoter_class_indices('/scratch/st-cdeboer-1/justin/data/test_subset_ids/yeast_seqs.csv')
    random = load_promoter_class_indices('/scratch/st-cdeboer-1/justin/data/test_subset_ids/all_random_seqs.csv')
    challenging = load_promoter_class_indices('/scratch/st-cdeboer-1/justin/data/test_subset_ids/challenging_seqs.csv')
    SNVs = load_promoter_class_indices('/scratch/st-cdeboer-1/justin/data/test_subset_ids/all_SNVs_seqs.csv')
    motif_perturbation = load_promoter_class_indices('/scratch/st-cdeboer-1/justin/data/test_subset_ids/motif_perturbation_seqs.csv')
    motif_tiling = load_promoter_class_indices('/scratch/st-cdeboer-1/justin/data/test_subset_ids/motif_tiling_seqs.csv')

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
    with open(path, 'w') as f:
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

def eval_human_model(al_method: str, round: int, arch: str, seed: int, batch_size: int=4096):
    test_path_ID = f"/scratch/st-cdeboer-1/justin/data/al_new/human/test.txt" 
    test_path_OOD = f"/scratch/st-cdeboer-1/justin/data/al_new/human/test_ood.txt" 
    test_path_SNV = f"/scratch/st-cdeboer-1/justin/data/al_new/human/test_snvs.txt" 
    seqsize = 200
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    '''
    model=load_model(species='human',
                    al_method=al_method,
                    arch=arch,
                    seed= seed,
                    round= round)
    '''
    model = init_model(species='human', arch=arch)
    path = f'/scratch/st-cdeboer-1/justin/models/new_codebase/100k_models_orig/human/{arch}_{seed}/model_best.pth'
    model.load_state_dict(torch.load(path, weights_only=True,map_location=device))
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

    #file=f"/scratch/st-cdeboer-1/justin/models/al_v2/human/round_{round}/{al_method}/{arch}_{seed}/results.txt"
    file = f'/scratch/st-cdeboer-1/justin/models/new_codebase/100k_models_orig/human/{arch}_{seed}/results.txt'

    with open(file, 'w') as f:
        f.write(f"ID\t{final_result[0]}\n")
        f.write(f"OOD\t{final_result[1]}\n")
        f.write(f"SNV\t{final_result[2]}\n")

def eval_yeast_model(al_method: str, round: int, arch: str, seed: int, batch_size: int=4096):
    test_path = f"/scratch/st-cdeboer-1/justin/data/al_new/yeast/test.txt" 
    seqsize = 150
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_dl = prepare_dataloader(test_path, 
                                seqsize=seqsize, 
                                species='yeast',
                                batch_size=batch_size,
                                shuffle=False)
    
    '''
    model=load_model(species='yeast',
            al_method=al_method,
            arch=arch,
            seed= seed,
            round= round)
    '''
    model = init_model(species='yeast', arch=arch)
    path = f'/scratch/st-cdeboer-1/justin/models/new_codebase/100k_models_orig/yeast/{arch}_{seed}/model_best.pth'
    model.load_state_dict(torch.load(path, weights_only=True,map_location=device))
    model.to(device).eval()

    with torch.inference_mode():
        predictions=[]
        for batch in test_dl:
            X = batch["x"].to(device)
            predictions.append(model(X).cpu().numpy())
    all_preds=np.concatenate(predictions,axis=0)
    all_preds=np.squeeze(all_preds)
    result = average_fwd_rev_pred(data=all_preds)

    #file=f"/scratch/st-cdeboer-1/justin/models/al_v2/yeast/round_{round}/{al_method}/{arch}_{seed}/results.txt"
    file = f'/scratch/st-cdeboer-1/justin/models/new_codebase/100k_models_orig/yeast/{arch}_{seed}/results.txt'
    evaluate_yeast_predictions(result,path=file)

def eval_model(species: str, 
               al_method: str, 
               round: int, 
               arch: str, 
               seed: int, 
               batch_size: int=4096):
    match species:
        case 'human':
            return eval_human_model(al_method=al_method,round=round,arch=arch,seed=seed,batch_size=batch_size)
        case 'yeast':
            return eval_yeast_model(al_method=al_method,round=round,arch=arch,seed=seed,batch_size=batch_size)
