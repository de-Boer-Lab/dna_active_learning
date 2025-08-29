import os, argparse
import pandas as pd

def update_train(original_train: str, selected: str, output_file: str):
    with open(output_file, "w") as f:
        for in_file in [original_train, selected]:
            with open(in_file, "r") as infile:
                f.write(infile.read())

def update_pool(original_pool:str, selected:str, output:str):
    df1=pd.read_csv(original_pool,sep='\t',header=None)
    df2=pd.read_csv(selected,sep='\t',header=None)
    df=df1[~df1[0].isin(df2[0])]
    df.to_csv(output,sep='\t', index=False,header=False)

def _update_train_and_pool(species: str,
                           next_round: int,
                           al_method: str,
                           arch: str,
                           seed: int):
    os.chdir(f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}")

    if next_round>1:
        update_train(f'round_{next_round-1}/{al_method}/{arch}_{seed}/train.txt',f"round_{next_round}/{al_method}/{arch}_{seed}/selected.txt",f"round_{next_round}/{al_method}/{arch}_{seed}/train.txt")
        if next_round < 3:
            update_pool(f'round_{next_round-1}/{al_method}/{arch}_{seed}/pool.txt',f"round_{next_round}/{al_method}/{arch}_{seed}/selected.txt",f"round_{next_round}/{al_method}/{arch}_{seed}/pool.txt")
        os.remove(f'round_{next_round-1}/{al_method}/{arch}_{seed}/pool.txt')
        os.remove(f'round_{next_round-1}/{al_method}/{arch}_{seed}/train.txt')

    else: # round 0->1
        update_train('round_0/common/train.txt',f"round_1/{al_method}/{arch}_{seed}/selected.txt",f"round_1/{al_method}/{arch}_{seed}/train.txt")
        update_pool('round_0/common/pool.txt',f"round_1/{al_method}/{arch}_{seed}/selected.txt",f"round_1/{al_method}/{arch}_{seed}/pool.txt")

    print("Done!")

'''
will make this look nicer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("next_round",type=int)
    parser.add_argument("--al_methods", nargs='+')
    args = parser.parse_args()

    os.chdir(f"/scratch/st-cdeboer-1/justin/data/al_v2/{args.species}")
    print("AL methods:",args.al_methods)

    if args.next_round>1:
        for method in ['mcd','kmeans','lcmd']:
            if method in args.al_methods:
                for arch in ['cnn','rnn','attn']:
                    for seed in range(1,6):
                        update_train(f'round_{args.next_round-1}/{method}/{arch}_{seed}/train.txt',f"round_{args.next_round}/{method}/{arch}_{seed}/selected.txt",f"round_{args.next_round}/{method}/{arch}_{seed}/train.txt")
                        if args.next_round < 3:
                            update_pool(f'round_{args.next_round-1}/{method}/{arch}_{seed}/pool.txt',f"round_{args.next_round}/{method}/{arch}_{seed}/selected.txt",f"round_{args.next_round}/{method}/{arch}_{seed}/pool.txt")
                        os.remove(f'round_{args.next_round-1}/{method}/{arch}_{seed}/pool.txt')
                        os.remove(f'round_{args.next_round-1}/{method}/{arch}_{seed}/train.txt')

        if 'same_arch' in args.al_methods:
            for arch in ['cnn','rnn','attn']:
                update_train(f'round_{args.next_round-1}/same_arch/{arch}/train.txt',f"round_{args.next_round}/same_arch/{arch}/selected.txt",f"round_{args.next_round}/same_arch/{arch}/train.txt")
                if args.next_round < 3:
                    update_pool(f'round_{args.next_round-1}/same_arch/{arch}/pool.txt',f"round_{args.next_round}/same_arch/{arch}/selected.txt",f"round_{args.next_round}/same_arch/{arch}/pool.txt")
                os.remove(f'round_{args.next_round-1}/same_arch/{arch}/pool.txt')
                os.remove(f'round_{args.next_round-1}/same_arch/{arch}/train.txt')
            
        for method in ['diff_arch','rnn-cnn']:
            if method in args.al_methods:
                for seed in range(1,6):
                    update_train(f'round_{args.next_round-1}/{method}/seed_{seed}/train.txt',f"round_{args.next_round}/{method}/seed_{seed}/selected.txt",f"round_{args.next_round}/{method}/seed_{seed}/train.txt")
                    if args.next_round < 3:
                        update_pool(f'round_{args.next_round-1}/{method}/seed_{seed}/pool.txt',f"round_{args.next_round}/{method}/seed_{seed}/selected.txt",f"round_{args.next_round}/{method}/seed_{seed}/pool.txt")
                    os.remove(f'round_{args.next_round-1}/{method}/seed_{seed}/pool.txt')
                    os.remove(f'round_{args.next_round-1}/{method}/seed_{seed}/train.txt')
    else: # round 0->1
        for method in ['mcd','kmeans','lcmd']:
            if method in args.al_methods:
                for arch in ['cnn','rnn','attn']:
                    for seed in range(1,6):
                        update_train('round_0/common/train.txt',f"round_1/{method}/{arch}_{seed}/selected.txt",f"round_1/{method}/{arch}_{seed}/train.txt")
                        update_pool('round_0/common/pool.txt',f"round_1/{method}/{arch}_{seed}/selected.txt",f"round_1/{method}/{arch}_{seed}/pool.txt")
        if 'same_arch' in args.al_methods:
            for arch in ['cnn','rnn','attn']:
                update_train('round_0/common/train.txt',f"round_1/same_arch/{arch}/selected.txt",f"round_1/same_arch/{arch}/train.txt")
                update_pool('round_0/common/pool.txt',f"round_1/same_arch/{arch}/selected.txt",f"round_1/same_arch/{arch}/pool.txt")
        for method in ['rnn-cnn','diff_arch']:
            if method in args.al_methods:
                for seed in range(1,6):
                    update_train('round_0/common/train.txt',f"round_1/{method}/seed_{seed}/selected.txt",f"round_1/{method}/seed_{seed}/train.txt")
                    update_pool('round_0/common/pool.txt',f"round_1/{method}/seed_{seed}/selected.txt",f"round_1/{method}/seed_{seed}/pool.txt")

    print("Done!")

if __name__ == "__main__":
    main()
'''