import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from torch import nn
import gc

try:
    import cupy as cp
except ImportError:
    pass

if TYPE_CHECKING:
    from nextfrag.path_resolver import PathResolver


def write_selections(
    cfg: "PathResolver",
    result_df: pd.DataFrame,
    symlinked_cfgs: Optional[List["PathResolver"]] = None,
):
    """Write selected sequences to cfg's data directory.

    For ensemble strategies, pass the non-canonical cfgs as symlinked_cfgs —
    each one's data_dir will be symlinked to cfg.data_dir so all arch/seed
    directories share the same selection.
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(cfg.selected_path, sep='\t', index=False, header=False, columns=[0, 1])

    if symlinked_cfgs:
        for other in symlinked_cfgs:
            link = other.data_dir
            if not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(cfg.data_dir, target_is_directory=True)


def _forward(model: nn.Module, X: torch.Tensor, use_predict: bool = None):
    if use_predict is None:
        use_predict = hasattr(model, 'predict')
    return model.predict(X) if use_predict else model(X)

def free_gpu_mem():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except NameError:
        pass

def free_gpu_mem_gc():
    gc.collect()
    free_gpu_mem()
