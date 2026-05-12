from pathlib import Path
from nextfrag.path_resolver import PathResolver


def update_train_and_pool(cfg: PathResolver, update_pool: bool = True):
    """Update training and pool data between active learning rounds.

    Reads selected.txt and the previous round's train/pool from cfg.prev_round(),
    writes new train.txt and pool.txt for the current round, and cleans up the
    previous round's files after round 1.

    Args:
        cfg: PathResolver for the round that just finished selection.
        update_pool: Whether to update pool (not updated on the last round).
    """
    prev = cfg.prev_round()
    prev_data_dir = prev.data_dir
    prev_train = prev_data_dir / "train.txt"
    prev_pool  = prev_data_dir / "pool.txt"

    curr_data_dir = cfg.data_dir
    curr_selected = cfg.selected_path
    curr_train    = curr_data_dir / "train.txt"
    curr_pool     = curr_data_dir / "pool.txt"
    curr_data_dir.mkdir(parents=True, exist_ok=True)

    _update_train(prev_train=prev_train, selected=curr_selected, curr_train=curr_train)
    if update_pool:
        _update_pool(prev_pool=prev_pool, selected=curr_selected, curr_pool=curr_pool)
    if cfg.round_num > 1:
        prev_train.unlink(missing_ok=True)
        prev_pool.unlink(missing_ok=True)

def _update_train(prev_train: str | Path,
                 selected: str | Path,
                 curr_train: str | Path):
    with open(curr_train, "w") as f:
        for in_file in [prev_train, selected]:
            with open(in_file, "r") as infile:
                f.write(infile.read())


def _update_pool(prev_pool: str | Path,
                selected: str | Path,
                curr_pool: str | Path):
    with open(selected, "r") as f:
        selected_seqs = set(f.readlines())

    with open(prev_pool, "r") as f1, open(curr_pool, "w") as f2:
        for line in f1:
            if line not in selected_seqs:
                f2.write(line)
