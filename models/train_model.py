import torch
import argparse
import json
from pathlib import Path
from sequence_selection.dataloader import prepare_dataloader
from sequence_selection.evaluation import eval_model
from .trainer import Trainer
from .model_loader import init_model
from nextFrag.config import get_project_root, DATASET_CONFIG, ARCH_CONFIG

def train_al_model(
    dataset: str,
    arch: str,
    al_strategy: str,
    round_num: int,
    seed: int,
    **kwargs
):
    """
    Train a model as part of an active learning experiment.
    
    Uses the standard AL directory structure and handles path construction automatically.
    
    Args:
        dataset: Dataset name ('yeast' and 'human' built-in)
        arch: Architecture name.  Either register in init_model() and
              ARCH_CONFIG, or supply model_class via **kwargs.
        al_strategy: Active learning strategy name
        round_num: AL round number
        seed: Random seed for reproducibility
        **kwargs: Forwarded to train_model() — see its docstring for options
    """
    project_root = get_project_root()
    experiment_path = project_root / dataset / f"round_{round_num}" / al_strategy / f"{arch}_{seed}"
    return train_model(
        dataset=dataset,
        arch=arch,
        train_path=experiment_path / "data" / "train.txt",
        val_path=project_root / dataset / "val.txt",
        model_dir=experiment_path / "model",
        results_path=experiment_path / "model" / "results.txt",
        seed=seed,
        **kwargs
    )


def train_model(
    dataset: str,
    arch: str,
    train_path: str | Path,
    val_path: str | Path,
    model_dir: str | Path,
    results_path: str | Path = None,
    seed: int = 42,
    num_epochs: int = None,
    train_batch_sz: int = None,
    val_batch_sz: int = 4096,
    lr: float = None,
    model_class=None,
    model_kwargs: dict = None,
):
    """Train a model with explicit data paths.

    For registered architectures, pass an ``arch`` name that is listed in 
    ``init_model()`` and ``ARCH_CONFIG``.  The model is instantiated 
    automatically.

    For arbitrary subclasses of nn.Module, pass ``model_class`` (type or 
    dotted import string) and optionally ``model_kwargs``.  ``arch`` is 
    then only used for directory naming in AL experiments, and ``lr`` 
    must be supplied explicitly.

    Args:        
        dataset: Dataset name ('yeast' or 'human') - used for defaults
        arch: Architecture name (registered) or any identifier string (custom)
        train_path: Path to training data file
        val_path: Path to validation data file
        model_dir: Directory to save model checkpoints
        results_path: Path to save evaluation results (defaults to model_dir/results.txt)
        seed: Random seed for reproducibility
        num_epochs: Training epochs (defaults from ARCH_CONFIG if arch is registered)
        train_batch_sz: Training batch size (defaults from DATASET_CONFIG)
        val_batch_sz: Validation batch size (default: 4096)
        lr: Learning rate (defaults from ARCH_CONFIG if arch is registered;
            required when using model_class)
        model_class: nn.Module subclass or dotted import path string.
                     When provided, bypasses the arch registry.
        model_kwargs: Keyword arguments forwarded to model_class()
    """
    model_dir = Path(model_dir)
    if results_path is None:
        results_path = model_dir / "results.txt"
    if train_batch_sz is None:
        train_batch_sz = DATASET_CONFIG[dataset]["batch_sz"]

    arch_cfg = ARCH_CONFIG.get(arch, {})
    if lr is None:
        if "lr" in arch_cfg:
            lr = arch_cfg["lr"]
        else:
            raise ValueError(
                f"lr must be specified for unregistered architecture '{arch}'.\n"
                "Either add it to ARCH_CONFIG in config.py, or pass lr= explicitly."
            )
    if num_epochs is None:
        num_epochs = arch_cfg.get("num_epochs", 80)

    if isinstance(model_class, str):
        model_class = import_class(model_class)

    generator = torch.Generator()
    generator.manual_seed(seed)

    model = init_model(
        dataset=dataset, arch=arch,
        model_class=model_class, model_kwargs=model_kwargs
    )

    seqsize = DATASET_CONFIG[dataset]["seqsize"]
    train_dl = prepare_dataloader(
        train_path,
        seqsize=seqsize,
        dset=dataset,
        batch_size=train_batch_sz,
        shuffle=True,
        generator=generator,
    )
    val_dl = prepare_dataloader(
        val_path,
        seqsize=seqsize,
        dset=dataset,
        batch_size=val_batch_sz,
        shuffle=False,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        model_dir=model_dir,
        num_epochs=num_epochs,
        lr=lr,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainer.fit()

    return eval_model(
        model_path=model_dir / "model_best.pth",
        out_file=results_path,
        dataset=dataset,
        arch=arch,
        model_class=model_class,
        model_kwargs=model_kwargs,
    )

def import_class(dotted_path: str) -> type:
    """Import a class from a dotted module path, e.g. ``'mypackage.models.MyModel'``."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── AL mode ──────────────────────────────────────────────────────────────
    al_parser = subparsers.add_parser("al", help="Train using AL experiment structure")
    al_parser.add_argument("dataset", choices=["yeast", "human"])
    al_parser.add_argument("arch", type=str)
    al_parser.add_argument("--strategy", type=str, required=True)
    al_parser.add_argument("--round", type=int, required=True)
    al_parser.add_argument("--seed", type=int, required=True)
    al_parser.add_argument("--epochs", type=int)
    al_parser.add_argument("--lr", type=float)
    al_parser.add_argument("--train-batch-size", type=int)
    al_parser.add_argument("--val-batch-size", type=int)
    al_parser.add_argument("--model-class", type=str)
    al_parser.add_argument("--model-kwargs", type=str, default="{}")

    # ── Custom mode ───────────────────────────────────────────────────────────
    custom_parser = subparsers.add_parser("custom", help="Train with explicit data paths")
    custom_parser.add_argument("dataset", choices=["yeast", "human"])
    custom_parser.add_argument("arch", type=str)
    custom_parser.add_argument("--train", type=str, required=True)
    custom_parser.add_argument("--val", type=str, required=True)
    custom_parser.add_argument("--model-dir", type=str, required=True)
    custom_parser.add_argument("--results", type=str)
    custom_parser.add_argument("--seed", type=int, default=42)
    custom_parser.add_argument("--epochs", type=int)
    custom_parser.add_argument("--lr", type=float)
    custom_parser.add_argument("--train-batch-size", type=int)
    custom_parser.add_argument("--val-batch-size", type=int)
    custom_parser.add_argument("--model-class", type=str)
    custom_parser.add_argument("--model-kwargs", type=str, default="{}")

    args = parser.parse_args()
    model_kwargs = json.loads(args.model_kwargs) or None

    if args.mode == "al":
        print(f"Training AL model: {args.dataset}/{args.arch}")
        print(f"  Strategy: {args.strategy}, Round: {args.round}, Seed: {args.seed}")
        return train_al_model(
            dataset=args.dataset,
            arch=args.arch,
            al_strategy=args.strategy,
            round_num=args.round,
            seed=args.seed,
            num_epochs=args.epochs,
            lr=args.lr,
            train_batch_sz=args.train_batch_size,
            val_batch_sz=args.val_batch_size,
            model_class=args.model_class,
            model_kwargs=model_kwargs,
        )
    else:
        print(f"Training custom model: {args.dataset}/{args.arch}")
        print(f"  Train: {args.train}")
        print(f"  Val: {args.val}")
        print(f"  Model dir: {args.model_dir}")
        return train_model(
            dataset=args.dataset,
            arch=args.arch,
            train_path=args.train,
            val_path=args.val,
            model_dir=args.model_dir,
            results_path=args.results,
            seed=args.seed,
            num_epochs=args.epochs,
            lr=args.lr,
            train_batch_sz=args.train_batch_size,
            val_batch_sz=args.val_batch_size,
            model_class=args.model_class,
            model_kwargs=model_kwargs,
        )

if __name__ == "__main__":
    main()
