import torch
import torch.nn as nn
from pathlib import Path
import importlib
from . import dream_models
from nextFrag.config import get_project_root, DATASET_CONFIG

def load_al_model(
    dataset: str,
    arch: str,
    al_strategy: str,
    round_num: int,
    seed: int,
    model_class: type = None,
    model_kwargs: dict = None,
) -> nn.Module:
    """Load checkpoint from a completed AL experiment.

    Constructs the path from the standard AL directory structure and delegates
    to ``load_model``.  This is the primary entry point for loading models
    within the AL workflow.
    """
    path = (get_project_root() / dataset / f"round_{round_num}"
        / al_strategy / f"{arch}_{seed}" / "model" / "model_best.pth")
    return load_model(path=path, dataset=dataset, arch=arch,
                      model_class=model_class, model_kwargs=model_kwargs)


def load_model(
    path: str | Path,
    dataset: str = None,
    arch: str = None,
    model_class: type = None,
    model_kwargs: dict = None,
) -> nn.Module:
    """Instantiate a model and load weights from an explicit checkpoint path.

    For loading from the standard AL directory structure, use
    ``load_al_model`` instead.

    For registered architectures (those in ``init_model``'s match statement
    and ``ARCH_CONFIG``), ``dataset`` and ``arch`` are sufficient.  For a
    one-off model not registered here, pass ``model_class`` (and optionally
    ``model_kwargs``) — ``dataset`` and ``arch`` are then not required.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = init_model(dataset=dataset, arch=arch,
                       model_class=model_class, model_kwargs=model_kwargs)

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state_dict)
    return model


def init_model(
    dataset: str = None,
    arch: str = None,
    model_class: type = None,
    model_kwargs: dict = None,
) -> nn.Module:
    """Instantiate a model without loading weights.

    Dispatches by ``arch`` name for registered architectures.  Pass
    ``model_class`` (and optionally ``model_kwargs``) to instantiate an
    arbitrary ``nn.Module`` directly, bypassing the registry.
    """
    if model_class is not None:
        return model_class(**(model_kwargs or {}))

    match arch:
        case "cnn" | "rnn" | "attn":
            return _init_dream_model(dataset, arch)

        # ── Register new architectures here ──────────────────────────────────
        # case "myarch":
        #     return _init_myarch_model(dataset)
        # ─────────────────────────────────────────────────────────────────────

        case _:
            raise ValueError(
                f"Unknown architecture '{arch}'.\n"
                f"To add it: define _init_{arch}_model() below, add a case here,\n"
                "and add its training defaults to ARCH_CONFIG in config.py.\n"
                "Alternatively, pass model_class= to use any nn.Module directly."
            )


def _init_dream_model(dataset: str, arch: str) -> nn.Module:
    seqsize = DATASET_CONFIG[dataset]["seqsize"]
    num_channels = DATASET_CONFIG[dataset]["in_channels"]
    match arch:
        case "rnn":
            return dream_models.DREAM_RNN(
                in_channels=num_channels, final_block=dataset, seqsize=seqsize
            )
        case "cnn":
            return dream_models.DREAM_CNN(
                in_channels=num_channels, final_block=dataset, seqsize=seqsize
            )
        case "attn":
            return dream_models.DREAM_ATTN(
                in_channels=num_channels, final_block=dataset, seqsize=seqsize
            )

# ── Template: adding a new architecture ──────────────────────────────────────
#
# 1. Define an init function here.
#
#   def _init_myarch_model(dataset: str) -> nn.Module:
#       seqsize = DATASET_CONFIG[dataset]["seqsize"]
#       num_channels = DATASET_CONFIG[dataset]["in_channels"]
#       return MyModel(in_channels=num_channels, seqsize=seqsize)
#
# 2. Register it in init_model()'s match statement above:
#
#       case "myarch":
#           return _init_myarch_model(dataset)
#
# 3. Add training defaults to ARCH_CONFIG in config.py:
#
#   ARCH_CONFIG = {
#       ...
#       "myarch": {"lr": 0.001, "num_epochs": 80},
#   }
#
# ─────────────────────────────────────────────────────────────────────────────
