import torch
import numpy as np
from scipy.stats import pearsonr
import csv
import dream_models
from dl_utils import prepare_dataloader

def init_model(species: str, arch: str):
    match species:
        case 'yeast':
            seqsize=150
            num_channels = 6
        case 'human':
            seqsize=200
            num_channels = 5

    match arch:
        case 'rnn':
            return dream_models.DREAM_RNN(in_channels=num_channels, seqsize=seqsize)
        case 'cnn':
            return dream_models.DREAM_CNN(in_channels=num_channels,seqsize=seqsize)
        case 'attn':
            return dream_models.DREAM_ATTN(in_channels=num_channels,seqsize=seqsize)

def load_model(species: str,
               al_method: str,
               arch: str,
               seed: int,
               round: int):

    model = init_model(species=species, arch=arch)
    path = f'/scratch/st-cdeboer-1/justin/models/al_v2/{species}/round_{round}/{al_method}/{arch}_{seed}/model_best.pth'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path, weights_only=True))
    else: # cpu
        model.load_state_dict(torch.load(path, weights_only=False,map_location=torch.device('cpu')))
    return model

def load_ground_truth(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
    expressions = [float(line[1]) for line in lines]
    return np.array(expressions)

def average_fwd_rev_pred(data: np.array) -> np.array:
    data=np.concatenate(data,axis=0)
    data=np.squeeze(data)
    num_samples=len(data)//2
    return (data[:num_samples]+data[num_samples:])/2

def eval_model(species: str, al_method: str, round: int, arch: str, seed: int, batch_size: int=4096):
    test_path = f"/scratch/st-cdeboer-1/justin/data/al_new/{species}/test.txt" 
    seqsize = 200 if species == 'human' else 150
    device=torch.device("cuda")

    test_dl = prepare_dataloader(test_path, 
                                seqsize=seqsize, 
                                species=species,
                                batch_size=batch_size,
                                shuffle=False)
        
    model=load_model(species=species,
            al_method=al_method,
            arch=arch,
            seed= seed,
            round= round)
    model.to(device).eval()

    with torch.inference_mode():
        predictions=[]
        for batch in test_dl:
            X = batch["x"].to(device)
            predictions.append(model.forward(X).cpu().numpy())
    
    result = average_fwd_rev_pred(data=predictions)

    file=f"/scratch/st-cdeboer-1/justin/models/al_v2/{species}/round_{round}/{al_method}/{arch}_{seed}/results.txt"
    gt=load_ground_truth(test_path)
    with open(file, 'w') as f:
        f.write(str(pearsonr(result,gt)[0]))
    #return evaluate_predictions(predictions,False,file)
    # will fix to eval for all test sets (subsets for yeast; {id ood snv} for human)
