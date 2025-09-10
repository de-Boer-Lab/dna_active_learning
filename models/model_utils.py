import torch
from pathlib import Path
from . import dream_models

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
               arch: str,
               path: str | Path = None, 
               al_method: str = None,
               seed: int = None,
               round: int = None):
    '''
    expects either a path to a model.pth file OR 
    can construct a path with al_method, seed, round
    '''

    model = init_model(species=species, arch=arch)
    if path is not None:
        filepath=path
    else: # infer from other params
        pass # e.g. /data_root/{species}/{round}/{al_method}/{arch}_{seed}/model_best.pth
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(filepath, weights_only=True))
    else: # cpu
        model.load_state_dict(torch.load(filepath, weights_only=False,map_location=torch.device('cpu')))
    return model
