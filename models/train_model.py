import torch, argparse
from pathlib import Path
from .dl_utils import prepare_dataloader
from .trainer import Trainer
from .model_utils import init_model
from .evaluation import eval_model

def train_model(species: str, 
                arch: str, 
                model_path: str | Path = None,
                al_method: str=None,
                round: int=None,
                seed: int=42,
                num_epochs: int = 80):
    train_path = f"data/{species}/demo_train.txt" # replace with actual path
    val_path = f"data/{species}/demo_val.txt" # replace with actual path
    seqsize = 200 if species == 'human' else 150
    train_batch_sz = 32
    valid_batch_sz = 4096
    lr = 0.001 if arch == 'attn' else 0.005

    if model_path is not None:
        model_dir = model_path 
    else:
        pass # infer from other args, e.g. /model_root/{species}/{round}/{al_method}/{arch}_{seed}

    generator = torch.Generator()
    generator.manual_seed(seed)

    model=init_model(species=species,arch=arch)

    train_dl = prepare_dataloader(train_path, 
                                seqsize=seqsize,
                                species=species,
                                batch_size=train_batch_sz,
                                shuffle = True,
                                generator=generator)
    val_dl = prepare_dataloader(val_path, 
                                seqsize=seqsize, 
                                species=species,
                                batch_size=valid_batch_sz,
                                shuffle=False)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        model_dir=model_dir,
        num_epochs=num_epochs,
        lr=lr,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    trainer.fit()

    eval_model(model_path=f"{model_dir}/model_best.pth",
               out_file= f"{model_dir}/results.txt",
               species=species,
               arch=arch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--al_method",type=str)
    parser.add_argument("--round",type=int)
    parser.add_argument("--seed",type=int)
    args = parser.parse_args()

    print("Received:")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")

    if args.model_path is not None:
        return train_model(species=args.species,
                           arch=args.arch,
                           model_path=args.model_path)
    else:
        return train_model(species=args.species,
                           arch=args.arch,
                           al_method=args.al_method,
                           round=args.round,
                           seed=args.seed)

if __name__ == "__main__":
    main()
