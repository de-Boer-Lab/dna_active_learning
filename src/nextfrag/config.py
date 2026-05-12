import os
from pathlib import Path

def _config_file(name: str) -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "nextfrag" / name

def get_project_root() -> Path:
    cfg = _config_file("root")
    if not cfg.is_file():
        raise RuntimeError(
            f"nextFrag project root is not configured. Run setup_dataset.sh "
            f"to scaffold a dataset; it will record the root at {cfg}."
        )
    root = cfg.read_text().strip()
    if not root:
        raise RuntimeError(f"recorded project root at {cfg} is empty.")
    return Path(root).expanduser().resolve()

def get_default_n_selected() -> int | None:
    """Return the configured default n_selected, or None if not set.

    When None, n_selected is always encoded in directory names (e.g. mcd_20k).
    When set to a value (e.g. 20000), runs using that value use the plain
    strategy name (e.g. mcd) and other values still get the _Xk suffix.
    Configure via setup_dataset.sh --n-selected or by writing a number to
    ~/.config/nextfrag/n_selected.
    """
    cfg = _config_file("n_selected")
    if not cfg.is_file():
        return None
    txt = cfg.read_text().strip()
    return int(txt) if txt else None

DATASET_CONFIG = {
    'yeast': {'seqsize': 150, 'in_channels': 6, 'batch_sz': 256},
    'human': {'seqsize': 200, 'in_channels': 5, 'batch_sz': 32},
}

ARCH_CONFIG = {
    'rnn':  {'lr': 0.005, 'num_epochs': 80},
    'cnn':  {'lr': 0.005, 'num_epochs': 80},
    'attn': {'lr': 0.001, 'num_epochs': 80},
}

DEFAULT_N_SELECTED = get_default_n_selected()