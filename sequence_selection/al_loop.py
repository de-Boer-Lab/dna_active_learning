'''
AL loop function for non-ensemble methods (MC dropout, k-means, LCMD)
'''

from models.train_model import train_model
from models.update_train_and_pool import _update_train_and_pool
from sequence_selection.mc_dropout import mc_dropout
from sequence_selection.kmeans import kmeans_al
from sequence_selection.lcmd import lcmd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("al_method", choices=['mcd','kmeans','lcmd'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("seed",type=int)
    parser.add_argument("--num_rounds","-r",type=int,default=3)
    parser.add_argument("--num_selected", "-n", type=int, default=20_000)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    match args.al_method:
        case 'mcd':
            al_method=mc_dropout
        case 'kmeans':
            al_method=kmeans_al
        case 'lcmd':
            al_method=lcmd

    if args.num_selected == 20000:
        method=args.al_method
    else:
        method=f"{args.al_method}_{args.num_selected//1000}k"

    for round in range(1,args.num_rounds+1):
        al_method(species=args.species, arch=args.arch, round=round, seed=args.seed, num_selected=args.num_selected)
        print("Sequence selection complete!")
        _update_train_and_pool(species=args.species, next_round=round, al_method=method, arch=args.arch, seed=args.seed)
        print("Train and pool sets updated!")
        train_model(species=args.species, arch=args.arch, al_method=method, round=round, seed=args.seed) 
        print(f"Model trained, round {round} done!")

if __name__=="__main__":
    main()
