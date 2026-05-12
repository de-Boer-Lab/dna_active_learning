import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Optional, Any
from tqdm import tqdm
import json

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        model_dir: str | Path,
        num_epochs: int = 80,
        lr: float = 5e-3,
        weight_decay: float = 0.01,
        device: torch.device = torch.device("cpu")
    ):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.use_predict = hasattr(self.model, 'predict')

        div_factor = 25.0
        min_lr = lr / div_factor

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=min_lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            div_factor=div_factor,
            steps_per_epoch=len(train_dataloader),
            epochs=num_epochs,
            pct_start=0.3,
            three_phase=False
        )
        self.regression_criterion = nn.MSELoss()
        self.classification_criterion = nn.KLDivLoss(reduction= "batchmean")
        self.best_pearson = -float("inf")

    def train_step(self, batch: dict[str, Any]) -> float:
        self.model.train()
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device).float()

        y_pred = self.model(x).squeeze()
        if "y_probs" in batch: # yeast, classification
            y_probs = batch["y_probs"].to(self.device)
            loss = self.classification_criterion(y_pred, y_probs)
        else: # regression
            y = batch["y"].to(self.device)
            loss = self.regression_criterion(y_pred, y.squeeze(-1))

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        return loss.item()

    def validate(self) -> dict[str, float]:
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).float()
                pred = self._forward(x).squeeze()

                y_true.append(y.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        metrics = {
            "MSE": float(np.mean((y_pred - y_true) ** 2)),
            "pearsonr": float(np.corrcoef(y_true, y_pred)[0, 1])
        }
        return metrics

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), self.model_dir / f"model_{name}.pth")
        torch.save(self.optimizer.state_dict(), self.model_dir / f"optimizer_{name}.pth")
        torch.save(self.scheduler.state_dict(), self.model_dir / f"scheduler_{name}.pth")

    def fit(self):
        for epoch in tqdm(range(1, self.num_epochs + 1)):
            epoch_losses = []
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch}"):
                loss = self.train_step(batch)
                epoch_losses.append(loss)

            with open(self.model_dir / "losses.json", "a") as f:
                json.dump({f"epoch_{epoch}": epoch_losses}, f)
                f.write("\n")

            if self.val_dataloader is not None:
                metrics = self.validate()
                with open(self.model_dir / "val_metrics.json", "a") as f:
                    json.dump({f"epoch_{epoch}": metrics}, f)
                    f.write("\n")
                print(metrics)

                if metrics["pearsonr"] > self.best_pearson:
                    self.best_pearson = metrics["pearsonr"]
                    self.save_model("best")        
        for file in ['optimizer_best.pth', 'scheduler_best.pth']:
            filepath = self.model_dir / file
            filepath.unlink(missing_ok=True)

    def _forward(self, X: torch.Tensor):
        return self.model.predict(X) if self.use_predict else self.model(X)
