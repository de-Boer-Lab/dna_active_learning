import torch
from .dream_models import DREAM_RNN, DREAM_CNN, DREAM_ATTN

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
            return DREAM_RNN(in_channels=num_channels, seqsize=seqsize)
        case 'cnn':
            return DREAM_CNN(in_channels=num_channels,seqsize=seqsize)
        case 'attn':
            return DREAM_ATTN(in_channels=num_channels,seqsize=seqsize)

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
