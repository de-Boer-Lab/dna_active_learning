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
    os.chdir(f"data/{species}") # change to your data root

    '''
    The below assumes a file structure of:
    /data_root/{species}/round_{round}/{al_method}/{arch}_{seed}
    containing: selected.txt, train.txt, pool.txt
    '''
    if next_round>1:
        update_train(f"round_{next_round-1}/{al_method}/{arch}_{seed}/train.txt",
                     f"round_{next_round}/{al_method}/{arch}_{seed}/selected.txt",
                     f"round_{next_round}/{al_method}/{arch}_{seed}/train.txt")
        if next_round < 3:
            update_pool(f"round_{next_round-1}/{al_method}/{arch}_{seed}/pool.txt",
                        f"round_{next_round}/{al_method}/{arch}_{seed}/selected.txt",
                        f"round_{next_round}/{al_method}/{arch}_{seed}/pool.txt")
        os.remove(f'round_{next_round-1}/{al_method}/{arch}_{seed}/pool.txt')
        os.remove(f'round_{next_round-1}/{al_method}/{arch}_{seed}/train.txt')

    else: # round 0->1
        update_train("round_0/common/train.txt",
                     f"round_1/{al_method}/{arch}_{seed}/selected.txt",
                     f"round_1/{al_method}/{arch}_{seed}/train.txt")
        update_pool("round_0/common/pool.txt",
                    f"round_1/{al_method}/{arch}_{seed}/selected.txt",
                    f"round_1/{al_method}/{arch}_{seed}/pool.txt")
        