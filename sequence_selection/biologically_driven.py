import torch
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import argparse
from tqdm import tqdm
from models.model_loader import load_al_model
from .dataloader import prepare_dataloader
from .utils import write_selections, _forward
from nextFrag.config import get_project_root, DATASET_CONFIG

def max_expression(
    dataset: str,
    arch: str,
    round_num: int,
    seed: int,
    num_selected: int,
    batch_size: int=2048,
    lowest: bool=False
):
    strategy='max_expr' if not lowest else 'min_expr'
    data_path=get_project_root() / dataset / f'round_{round_num}' / strategy / f'{arch}_{seed}' / 'data' / 'pool.txt'
    seqsize=DATASET_CONFIG[dataset]['seqsize']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataloader=prepare_dataloader(
        data_path,
        seqsize=seqsize,
        dset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model=load_al_model(dataset=dataset,arch=arch,al_strategy=strategy,seed=seed,round_num=round_num)
    model.to(device).eval()
    
    df=pd.read_csv(data_path,header=None,sep='\t')
    num_seqs=len(df)

    all_preds=[]
    with torch.inference_mode():
        for batch in dataloader:
            X=batch['x'].to(device)
            pred=_forward(model, X)
            all_preds.append(pred.cpu().numpy())
    all_preds=np.concatenate(all_preds)
    all_preds=all_preds.reshape(2,num_seqs)
    all_preds=np.sum(all_preds,axis=0)/2 

    df['pred']=all_preds
    df=df.sort_values(by=['pred'],ascending=lowest).head(num_selected)
    write_selections(
        df, 
        dataset=dataset,
        strategy=strategy,
        round_num=round_num,
        num_selected=num_selected,
        arch=arch,
        seed=seed
    )

def ism(
    file_path: str | Path, 
    out_path: str | Path, 
    dataset: str, 
    job_id: int, 
    seqs_per_job: int=500_000,
    window_sz: int=6, 
    arch: str='rnn', 
    seed: int=1
):
    df = pd.read_csv(file_path, header=None, sep='\t')
    df.columns=['seq','expr']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dataset=='human':
        batch_size=6
        start_pos=0
        end_pos=200
        seqsize=DATASET_CONFIG['human']['seqsize']
        df= df[df['seq'].str.len()==seqsize]
    else: # yeast
        batch_size=16
        start_pos=57
        end_pos=137
        seqsize=DATASET_CONFIG['yeast']['seqsize']
        left_flank="AGTGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC"
        df= df[df['seq'].str[:len(left_flank)]==left_flank]
        df= df[~df['seq'].str.contains('N')]
    df = df.iloc[job_id*seqs_per_job:(job_id+1)*seqs_per_job]

    model=load_al_model(dataset=dataset,arch=arch,al_strategy='common',seed=seed,round='0')
    model.to(device).eval()

    attrs = np.empty((len(df), end_pos - start_pos, 3), dtype=np.float32)
    idx=0
    buffer = []
    for row in tqdm(df.itertuples()):
        seq = row.seq.upper()
        X=seq2tensor(seq,dataset)
        buffer.append(X)
        if len(buffer)>=batch_size:
            X=torch.cat(buffer,dim=0)
            y, y_ism = saturation_mutagenesis(
                model=model,
                X=X,
                start=start_pos,
                end=end_pos,
                device=device
            )
            
            y_attr = y_ism - y[:, None, None] 
            n=len(buffer)
            attrs[idx:idx+n] = y_attr.squeeze(-1).cpu().numpy()
            idx += n
            buffer=[]
    if buffer:
        X=torch.cat(buffer,dim=0)
        y, y_ism = saturation_mutagenesis(
            model=model,
            X=X,
            start=start_pos,
            end=end_pos,
            device=device
        )
        
        y_attr = y_ism - y[:, None, None] 
        n=len(buffer)
        attrs[idx:idx+n] = y_attr.squeeze(-1).cpu().numpy()
    
    result=compute_attributions(attrs,window_sz)
    out_df = pd.concat([df[["seq"]].reset_index(drop=True),
                        pd.DataFrame(result,columns=['mean','max','window max'])],axis=1)
    out_df.to_csv(out_path,sep='\t',index=None)

def _edit_distance_one(X, start, end): # X.shape==(C,L)
    if end < 0:
        end = X.shape[-1] + end + 1
    X_ = X.repeat((end-start)*3, 1, 1)

    coords = itertools.product(range(start, end),range(4))
    _next=0
    for pos, mut in coords:
        if X[mut,pos]==1: # no mutation
            continue
        X_[_next, :4, pos] = 0
        X_[_next, mut, pos] = 1
        _next+=1
    return X_

def saturation_mutagenesis(model, X, start=0, end=-1, device='cuda'):
    N, C, L = X.shape
    if end < 0:
        end = L + end + 1
    
    X_mut = [_edit_distance_one(X[i], start, end) for i in range(N)]
    X_mut = torch.cat(X_mut, dim=0)   # (N * n_mut, C, L)
    X_all = torch.cat((X,X_mut),dim=0)
    
    model = model.to(device).eval()
    dtype = next(model.parameters(), X).dtype
    
    with torch.inference_mode():
        y = _forward(model, X_all.to(device).type(dtype))

    y0 = y[:N]
    y_hat = y[N:].view(N,end-start, 3, *y.shape[1:])
    return y0, y_hat

def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'T': [0, 0, 0, 1]}
    return [mapping[base] for base in seq]

def seq2tensor(seq,dataset):
    ohe_seq = one_hot_encode(seq)
    rev_values = [0] * len(ohe_seq)
    is_singletons = [0] * len(ohe_seq)
    if dataset=='yeast':
        encoded = [list(ohe) + [rev] + [is_singleton] 
                    for ohe, rev, is_singleton in zip(ohe_seq, rev_values, is_singletons)]
    else:
        encoded = [list(ohe) + [rev] for ohe, rev in zip(ohe_seq, rev_values)]
    X = torch.Tensor(encoded).type(torch.float32)
    X = X.unsqueeze(0)
    X = torch.transpose(X, 2, 1)
    return X
def compute_attributions(attrs,window_sz):
    attrs=-attrs.mean(axis=2) # average across substitutions
    abs_attrs=np.abs(attrs)
    N,L=attrs.shape
    
    _mean=abs_attrs.mean(axis=1)
    maxpos=abs_attrs.argmax(axis=1)
    _max=attrs[np.arange(N),maxpos]
    
    windows = np.lib.stride_tricks.sliding_window_view(attrs, window_sz, axis=1)
    window_sums = windows.sum(axis=2)
    idx = np.abs(window_sums).argmax(axis=1)
    max_window = window_sums[np.arange(N), idx]

    result=np.stack((_mean,_max,max_window),axis=-1)
    return result
