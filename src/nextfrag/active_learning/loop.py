"""
Active Learning Loop
====================
Main training loop for non-ensemble AL strategies (MC Dropout, k-means, LCMD).

The loop consists of three steps per round:
1. Select sequences using the AL strategy
2. Update training and pool datasets
3. Train a new model on the updated data
"""

import argparse
from nextfrag.path_resolver import PathResolver
from nextfrag.utils import free_gpu_mem_gc
from nextfrag.config import DEFAULT_N_SELECTED
from nextfrag.models.train_model import train_al_model
from nextfrag.acquisition.mc_dropout import mc_dropout
from nextfrag.acquisition.diversity import diversity_al
from .utils import update_train_and_pool


def loop(
    cfg: PathResolver,
    num_rounds: int = 3,
    start_round: int = 1,
    train_initial_model: bool = False,
    acq_kwargs: dict | None = None,
    train_kwargs: dict | None = None,
):
    """Run the active learning loop for a given strategy.

    Args:
        cfg: PathResolver describing the run (dataset, acquisition, arch, seed,
             n_selected).
        num_rounds: Number of AL rounds to run.
        start_round: Round to start from; set > 1 to resume a partial run.
        train_initial_model: If True, train a round-0 model before the loop.
        acq_kwargs: Extra keyword arguments forwarded to the acquisition function.
        train_kwargs: Extra keyword arguments forwarded to train_al_model.
    """
    if cfg.acquisition == 'mcd':
        acquisition_fn = mc_dropout
    else:
        acquisition_fn = diversity_al
    
    final_round = start_round + num_rounds - 1

    print("=" * 40)
    print(f"Active Learning Loop - {cfg.acquisition.upper()}")
    print("=" * 40)
    print(f"Dataset:        {cfg.dataset}")
    print(f"Architecture:   {cfg.arch}")
    print(f"Acquisition:    {cfg.acquisition}")
    print(f"Seed:           {cfg.seed}")
    print(f"Rounds:         {start_round} to {final_round}")
    print(f"Selected/round: {cfg.n_selected}")
    print("=" * 40)
    
    if train_initial_model:
        print("Training initial model...")
        train_al_model(cfg.prev_round(), **(train_kwargs or {}))
        print("Model trained")
        print("=" * 40)

    for round_num in range(start_round, final_round + 1):
        print(f"ROUND {round_num}/{final_round}")
        print(f"\n[1/3] Selecting sequences using {cfg.acquisition}...")
        acquisition_fn(cfg, **(acq_kwargs or {}))
        print(f"Selected {cfg.n_selected:,} sequences")

        print(f"\n[2/3] Updating train and pool datasets...")
        update_train_and_pool(cfg, update_pool=(round_num != final_round))
        print("Datasets updated")

        free_gpu_mem_gc()

        print(f"\n[3/3] Training model...")
        train_al_model(cfg, **(train_kwargs or {}))
        print("Model trained")

        free_gpu_mem_gc()

        print(f"✓ Round {round_num} complete!")
        print("=" * 40)
        cfg=cfg.next_round()

    print(f"\n{'=' * 40}")
    print("Active Learning Loop Complete!")
    print(f"{'=' * 40}")
    print(f"Completed {num_rounds} rounds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=['yeast', 'human'], required=True)
    parser.add_argument("--acquisition", "-a", choices=['mcd', 'kmeans', 'lcmd'], required=True)
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--seed", "-s", type=int, required=True)
    parser.add_argument("--train-initial-model", action='store_true')
    parser.add_argument("--num-rounds", "-r", type=int, default=3)
    parser.add_argument("--n-selected", "-n", type=int, default=DEFAULT_N_SELECTED)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--train-batch-size", "-t", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--inference-batch-size", "-i", type=int, default=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.start_round < 1:
        parser.error("--start-round must be >= 1")
    if args.n_selected <= 0:
        parser.error("--n-selected must be positive")

    cfg = PathResolver(
        dataset=args.dataset,
        round_num=args.start_round,
        arch=args.arch,
        seed=args.seed,
        acquisition=args.acquisition,
        n_selected=args.n_selected,
    )

    train_kwargs = {}
    acq_kwargs = {}
    if hasattr(args, 'train_batch_size'):
        train_kwargs['train_batch_sz'] = args.train_batch_size

    if hasattr(args, 'inference_batch_size'):
        acq_kwargs['batch_size'] = args.inference_batch_size
        train_kwargs['val_batch_sz'] = args.inference_batch_size

    return loop(cfg, num_rounds=args.num_rounds, start_round=args.start_round,
                train_initial_model=args.train_initial_model,
                acq_kwargs=acq_kwargs or None, train_kwargs=train_kwargs or None)


if __name__ == "__main__":
    main()
