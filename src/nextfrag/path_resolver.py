from dataclasses import dataclass, replace
from pathlib import Path
from nextfrag.config import get_project_root, get_default_n_selected

_DEFAULT_N_SELECTED = get_default_n_selected()


@dataclass(frozen=True)
class PathResolver:
    """Identifies a single (dataset, round, acquisition, arch, seed) run and
    resolves all filesystem paths for that run.

    Round-0 layout (shared data, per-model checkpoints):
        {dataset}/round_0/data/           ← train.txt, pool.txt
        {dataset}/round_0/{arch}_{seed}/model/  ← model_best.pth

    Round-1+ layout:
        {dataset}/round_{n}/{acquisition_dir}/{arch}_{seed}/data/
        {dataset}/round_{n}/{acquisition_dir}/{arch}_{seed}/model/

    The acquisition directory name encodes n_selected when it differs from the
    configured default (see acquisition_dir property).
    """

    dataset: str
    round_num: int
    arch: str
    seed: int
    acquisition: str | None = None 
    n_selected: int | None = _DEFAULT_N_SELECTED

    def __post_init__(self):
        if self.round_num > 0 and self.acquisition is None:
            raise ValueError(f"acquisition required for round_num={self.round_num}")
        if self.n_selected is None and self.round_num > 0:
            raise ValueError(f"n_selected is required for round_num={self.round_num}")

    @property
    def acquisition_dir(self) -> str:
        """Strategy directory name. If DEFAULT_N_SELECTED is set and n_selected
        matches it: plain name. Otherwise, append _{n}k (rounded) or _{n} for
        values that aren't a whole number of thousands.
        """
        if _DEFAULT_N_SELECTED is not None and self.n_selected == _DEFAULT_N_SELECTED:
            return self.acquisition
        if self.n_selected % 1000 == 0:
            return f"{self.acquisition}_{self.n_selected // 1000}k"
        return f"{self.acquisition}_{self.n_selected}"

    @property
    def project_root(self) -> Path:
        return get_project_root()

    @property
    def dataset_root(self) -> Path:
        return self.project_root / self.dataset

    @property
    def round_root(self) -> Path:
        if self.round_num == 0:
            return self.dataset_root / "round_0"
        return (
            self.dataset_root
            / f"round_{self.round_num}"
            / self.acquisition_dir
            / f"{self.arch}_{self.seed}"
        )

    @property
    def data_dir(self) -> Path:
        return self.round_root / "data"

    @property
    def model_dir(self) -> Path:
        if self.round_num == 0:
            return self.round_root / f"{self.arch}_{self.seed}" / "model"
        return self.round_root / "model"

    @property
    def train_path(self) -> Path:
        return self.data_dir / "train.txt"

    @property
    def pool_path(self) -> Path:
        return self.data_dir / "pool.txt"

    @property
    def val_path(self) -> Path:
        return self.dataset_root / "val.txt"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model_best.pth"

    @property
    def selected_path(self) -> Path:
        if self.round_num == 0:
            raise ValueError("round 0 has no selected.txt")
        return self.data_dir / "selected.txt"

    @property
    def results_path(self) -> Path:
        """Shared results TSV for this dataset (created by setup_dataset.sh)."""
        return self.dataset_root / "results.tsv"

    def prev_round(self) -> "PathResolver":
        if self.round_num == 0:
            raise ValueError("no previous round before round 0")
        return replace(
            self,
            round_num=self.round_num - 1,
        )

    def next_round(self, *, acquisition: str | None = None) -> "PathResolver":
        if self.round_num == 0 and self.acquisition is None and acquisition is None:
            raise ValueError("acquisition required when advancing from round 0")
        return replace(
            self,
            round_num=self.round_num + 1,
            acquisition=acquisition or self.acquisition,
        )
