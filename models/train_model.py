import torch, argparse
from .dl_utils import prepare_dataloader
from .trainer import Trainer
from .model_utils import init_model
from .evaluation import eval_model

def train_model(species: str, 
                arch: str, 
                al_method: str,
                round: int,
                seed: int,
                num_epochs: int = 80):
    train_path = f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}/round_{round}/{al_method}/{arch}_{seed}/train.txt"
    val_path = f"/scratch/st-cdeboer-1/justin/data/al_v2/{species}/val.txt"
    model_dir = f"/scratch/st-cdeboer-1/justin/models/al_v2/{species}/round_{round}/{al_method}/{arch}_{seed}"
    seqsize = 200 if species == 'human' else 150
    train_batch_sz = 1024
    valid_batch_sz = 4096
    lr = 0.001 if arch == 'attn' else 0.005

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

    eval_model(species=species,al_method=al_method,round=round,arch=arch,seed=seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species",choices=['yeast','human'])
    parser.add_argument("arch",choices=['cnn', 'rnn', 'attn'])
    parser.add_argument("al_method",type=str)
    parser.add_argument("round",type=int)
    parser.add_argument("seed",type=int)
    args = parser.parse_args()

    print(f"received: {args.species} {args.arch} {args.al_method} {args.round} {args.seed}")
    train_model(species=args.species,
                 arch=args.arch,
                 al_method=args.al_method,
                 round=args.round,
                 seed=args.seed)

if __name__ == "__main__":
    main()
