import numpy as np
import torch, os, argparse
from models.model_utils import load_model
from models.dl_utils import prepare_dataloader, revcomp

def ensemble_diff_arch(species: str,
                       composition: str,
                       round: int,
                       seed: int,
                       num_selected: int):
    data_path = f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}/round_{round-1}/{composition}/{arch}_{seed}/pool.txt"
    seqsize = 200 if species == 'human' else 150
    batch_size = 4096
    device=torch.device("cuda")

    dataloader = prepare_dataloader(data_path, 
                                seqsize=seqsize,
                                species=species,
                                batch_size=batch_size,
                                shuffle = False)

    cnn=load_model(species=species,al_method=composition,arch='cnn',seed=seed,round=round-1)
    cnn = cnn.to(device).eval()

    rnn=load_model(species=species,al_method=composition,arch='rnn',seed=seed,round=round-1)
    rnn = rnn.to(device).eval()

    attn=load_model(species=species,al_method=composition,arch='attn',seed=seed,round=round-1)
    attn = attn.to(device).eval()

    with torch.inference_mode():
        seq2var={}
        var2seq={}
        seq2expr={}
        for batch in dataloader:
            X = batch["x"].to(device)
            y = batch["y"]

            if composition != 'cnn-attn':
                rnn_pred = rnn.forward(X)
            if composition != 'rnn-attn':
                cnn_pred = cnn.forward(X)
            if composition != 'rnn-cnn':
                attn_pred = attn.forward(X)

            if composition == 'all_arch':
                combined=torch.stack((cnn_pred,rnn_pred,attn_pred),dim=0)
                combined=combined.cpu().numpy()
                var = np.var(combined,axis=0)
            elif composition == 'rnn-cnn':
                var = torch.abs(rnn_pred-cnn_pred).cpu().numpy()
            elif composition == 'rnn-attn':
                var = torch.abs(rnn_pred-attn_pred).cpu().numpy()
            else: # cnn-attn
                var = torch.abs(cnn_pred-attn_pred).cpu().numpy()

            for i in range(var.size):
                if X[i][4,1].item()==0.0:
                    seq2expr.update({seq[i]:y[i].item()})
                else:
                    seq[i] = revcomp(seq[i])
                if (seq[i] not in seq2var) or (seq2var[seq[i]] < var[i]):
                    seq2var.update({seq[i]:var[i].item()})
        for seq in seq2var.keys():
            var2seq.update({seq2var[seq]:seq})

    keys = list(var2seq.keys())
    keys.sort(reverse=True)
    print(keys[:50])

    if num_selected == 20000:
        folder_name = composition
    else:
        n_selected=num_selected//1000
        folder_name = f"{composition}_{n_selected}k"

    out_path = f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}/round_{round}/{folder_name}/seed_{seed}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file = f'{out_path}/selected.txt'
    for arch in ['cnn','rnn','attn']:
        os.symlink(out_path, f'/scratch/st-cdeboer-1/justin/data/al_v2/{species}/round_{round}/{folder_name}/{arch}_{seed}')

    with open(file,'w') as f:
        for i in range(num_selected):
            seq = var2seq.get(keys[i])
            f.write(seq+'\t'+str(seq2expr.get(seq))+'\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("composition",choices=['all_arch','rnn-cnn','rnn-attn','cnn-attn'])
    parser.add_argument("round",type=int)
    parser.add_argument("seed",type=int)
    parser.add_argument("--num_selected",type=int,default=20000)
    args = parser.parse_args()
    
    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")

    ensemble_diff_arch(species=args.species,
                       composition=args.composition,
                       round=args.round,
                       seed=args.seed,
                       num_selected=args.num_selected)

if __name__ == "__main__":
    main()