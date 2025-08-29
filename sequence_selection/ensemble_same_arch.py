import numpy as np
import torch, os, argparse, copy
from models.model_utils import load_model
from models.dl_utils import prepare_dataloader, revcomp
from typing import List

def ensemble_same_arch(species: str,
                       arch: str,
                       round: int,
                       seeds: List[int],
                       num_selected: int):
    data_path = f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}/round_{round-1}/same_arch/{arch}_1/pool.txt"
    seqsize = 200 if species == 'human' else 150
    batch_size = 4096
    device=torch.device("cuda")

    dataloader = prepare_dataloader(data_path, 
                                seqsize=seqsize,
                                species=species,
                                batch_size=batch_size,
                                shuffle = False)

    models=[]
    for seed in seeds:
        model=load_model(species=species, al_method='same_arch', arch=arch, seed=seed, round=round-1)
        models.append(copy.deepcopy(model))  

    for model in models:
        model = model.to(device)
        model = model.eval()

    with torch.inference_mode():
        seq2var={}
        var2seq={}
        seq2expr={}
        for batch in dataloader:
            X = batch["x"].to(device)
            y = batch["y"]

            model_preds=[]
            for model in models:
                model_preds.append(model.forward(X))

            combined = torch.stack(model_preds).cpu().numpy()
            var = np.var(combined,axis=0)

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
        folder_name = 'same_arch'
    else:
        n_selected=num_selected//1000
        folder_name = f"same_arch_{n_selected}k"

    out_path = f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}/round_{round}/{folder_name}/{arch}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file = f'{out_path}/selected.txt'
    for seed in ['1','2','3','4','5']:
        os.symlink(out_path, f'{out_path}_{seed}')

    with open(file,'w') as f:
        for i in range(num_selected):
            seq = var2seq.get(keys[i])
            f.write(seq+'\t'+str(seq2expr.get(seq))+'\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("round",type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5])
    parser.add_argument("--num_selected",type=int,default=20000)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")

    ensemble_same_arch(species=args.species,
                       arch=args.arch,
                       round=args.round,
                       seeds=args.seeds,
                       num_selected=args.num_selected)

if __name__ == "__main__":
    main()