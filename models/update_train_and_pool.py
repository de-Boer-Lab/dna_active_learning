import os

def update_train(original_train: str, selected: str, output_file: str):
    with open(output_file, "w") as f:
        for in_file in [original_train, selected]:
            with open(in_file, "r") as infile:
                f.write(infile.read())

def update_pool(original_pool:str, selected:str, output:str):
    with open(selected, "r") as f:
        selected_seqs = set(f.readlines()) 

    with open(original_pool, "r") as f1, open(output, "w") as f2:
        for line in f1:
            if line not in selected_seqs:  
                f2.write(line)

def _update_train_and_pool(species: str,
                           next_round: int,
                           al_method: str,
                           arch: str,
                           seed: int,
                           num_rounds: int=3):
    '''
    This function assumes a file structure of:
    /data_root/{species}/round_{round}/{al_method}/{arch}_{seed}
    containing: selected.txt, train.txt, pool.txt
    '''
    os.chdir(f"/data_root/{species}") # change to your data root
    path = f"{al_method}/{arch}_{seed}"
    
    if next_round>1:
        update_train(original_train=f"round_{next_round-1}/{path}/train.txt",
                     selected=f"round_{next_round}/{path}/selected.txt",
                     output_file=f"round_{next_round}/{path}/train.txt")
        if next_round < num_rounds:
            update_pool(original_pool=f"round_{next_round-1}/{path}/pool.txt",
                        selected=f"round_{next_round}/{path}/selected.txt",
                        output=f"round_{next_round}/{path}/pool.txt")
        os.remove(f'round_{next_round-1}/{path}/pool.txt')
        os.remove(f'round_{next_round-1}/{path}/train.txt')

    else: # round 0->1
        update_train(original_train="round_0/common/train.txt",
                     selected=f"round_1/{path}/selected.txt",
                     output_file=f"round_1/{path}/train.txt")
        if num_rounds>1:
            update_pool(original_pool="round_0/common/pool.txt",
                    selected=f"round_1/{path}/selected.txt",
                    output=f"round_1/{path}/pool.txt")
        