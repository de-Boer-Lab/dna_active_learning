import torch
import argparse
import json
from pathlib import Path
import importlib
from .model_loader import init_model
from .evaluation import eval_model, eval_al_model
from .trainer import Trainer
from nextfrag.data.dataloader import build_dataloader
from nextfrag.config import DATASET_CONFIG, ARCH_CONFIG, DEFAULT_N_SELECTED
from nextfrag.path_resolver import PathResolver


def train_al_model(cfg: PathResolver, **kwargs):
    """Train a model as part of an active learning experiment.

    Constructs all paths from cfg and forwards training options via **kwargs.
    After training, evaluates the model and appends one row to the dataset-level
    results.tsv (created by setup_dataset.sh).

    Args:
        cfg: PathResolver for this round — supplies dataset, arch, seed, and
             all path properties.
        **kwargs: Forwarded to train_model() — see its docstring for options.
    """
    return train_model(
        dataset=cfg.dataset,
        arch=cfg.arch,
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        model_dir=cfg.model_dir,
        seed=cfg.seed,
        cfg=cfg,
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
    cfg: PathResolver = None,
    num_epochs: int = None,
    train_batch_sz: int = None,
    val_batch_sz: int = None,
    lr: float = None,
    model_class=None,
    model_kwargs: dict = None,
) -> dict | None:
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
        results_path: If provided, also write a human-readable results summary
                      there. AL workflows use the dataset-level results.tsv instead.
        seed: Random seed for reproducibility
        num_epochs: Training epochs (defaults from ARCH_CONFIG if arch is registered)
        train_batch_sz: Training batch size (defaults from DATASET_CONFIG)
        val_batch_sz: Validation batch size (default: 4096)
        lr: Learning rate (defaults from ARCH_CONFIG if arch is registered;
            required when using model_class)
        model_class: nn.Module subclass or dotted import path string.
                     When provided, bypasses the arch registry.
        model_kwargs: Keyword arguments forwarded to model_class()

    Returns:
        dict of metric_name → value from evaluation, or None if evaluation fails.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if train_batch_sz is None:
        train_batch_sz = DATASET_CONFIG[dataset]["batch_sz"]
    if val_batch_sz is None:
        val_batch_sz = 4096

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
    train_dl = build_dataloader(
        train_path,
        seqsize=seqsize,
        dataset=dataset,
        batch_size=train_batch_sz,
        shuffle=True,
        generator=generator,
    )
    val_dl = build_dataloader(
        val_path,
        seqsize=seqsize,
        dataset=dataset,
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

    if cfg is not None:
        return eval_al_model(
            cfg=cfg,
            batch_size=val_batch_sz,
            model_class=model_class,
            model_kwargs=model_kwargs,
        )
    if results_path is None:
        results_path = model_dir / 'results.tsv'
    return eval_model(
        model_path=model_dir / "model_best.pth",
        dataset=dataset,
        arch=arch,
        out_file=results_path,
        batch_size=val_batch_sz,
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
    al_parser.add_argument("--dataset", choices=["yeast", "human"], required=True)
    al_parser.add_argument("--arch", type=str, required=True)
    al_parser.add_argument("--acquisition", type=str, required=True)
    al_parser.add_argument("--round", type=int, required=True)
    al_parser.add_argument("--seed", type=int, required=True)
    al_parser.add_argument("--n-selected", type=int, default=DEFAULT_N_SELECTED)
    al_parser.add_argument("--epochs", type=int)
    al_parser.add_argument("--lr", type=float)
    al_parser.add_argument("--train-batch-size", type=int)
    al_parser.add_argument("--val-batch-size", type=int)
    al_parser.add_argument("--model-class", type=str)
    al_parser.add_argument("--model-kwargs", type=str, default="{}")

    # ── Custom mode ───────────────────────────────────────────────────────────
    custom_parser = subparsers.add_parser("custom", help="Train with explicit data paths")
    custom_parser.add_argument("--dataset", choices=["yeast", "human"], required=True)
    custom_parser.add_argument("--arch", type=str, required=True)
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
        cfg = PathResolver(
            dataset=args.dataset,
            round_num=args.round,
            arch=args.arch,
            seed=args.seed,
            acquisition=args.acquisition,
            n_selected=args.n_selected,
        )
        print(f"Training AL model: {args.dataset}/{args.arch}")
        print(f"  Acquisition: {args.acquisition}, Round: {args.round}, Seed: {args.seed}")
        return train_al_model(
            cfg,
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
