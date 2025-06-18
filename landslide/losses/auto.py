import re
from typing import Tuple

import torch
import torch.nn as nn

from .dice import BinaryDiceLoss
from .focal import BinaryFocalLossWithLogits
from .lovasz import LovaszHingeLoss


class AutoCriterion(nn.Module):
    def __init__(self, name: str, model: nn.Module, hyp, data, device):
        super().__init__()

        names, weights, instances = self.parse_criterion(name, model, hyp, data, device)
        self.losses = nn.ModuleList(instances)
        self.weights = weights
        self.names = names

    def __len__(self):
        return len(self.losses)

    def parse_criterion(self, criterion: str, model, hyp, data, device) -> nn.Module:
        entries = criterion.split("+")
        names = []
        weights = []
        functions = []

        for entry in entries:
            weight, name = self.parse_entry(entry)
            loss_fn = self.parse_instance(name, model, hyp, data, device)

            names.append(name)
            functions.append(loss_fn)
            weights.append(weight)
        return names, weights, functions

    def parse_entry(self, entry: str) -> Tuple[float, str]:
        """
        Parse a criterion entry with an optional weight.

        Expected format:
            "[weight]loss_name" or simply "loss_name"
        Extra spaces around the weight and name are allowed.

        Returns:
            A tuple (weight, name) where weight is a float and name is the loss name string.
        """
        pattern = r"^\s*(?:\[(?P<weight>[\d\.]+)\])?\s*(?P<name>\S.*)$"
        match = re.match(pattern, entry)
        if not match:
            raise ValueError(f"Could not parse criterion entry: {entry}")

        weight_str = match.group("weight")
        name = match.group("name").strip()
        weight = float(weight_str) if weight_str is not None else 1.0
        return weight, name

    def parse_instance(self, name: str, model, hyp, data, device) -> nn.Module:
        nc = data.get("nc", 1)
        if name == "binary_cross_entropy" or name == "bce":
            return nn.BCEWithLogitsLoss()
        elif name == "focal_loss":
            return BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0)
        elif name == "lovasz_hinge_loss" or name == "lovasz_loss" or name == "lovasz":
            return LovaszHingeLoss()
        elif name == "weighted_binary_cross_entropy" or name == "wbce":
            pos_weight = torch.tensor(data["pos_weights"]).reshape(nc, 1, 1).to(device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif name == "dice_loss" or name == "dice":
            return BinaryDiceLoss()
        else:
            print(f"[WARN] Unknown criterion: {name}, using BCEWithLogitsLoss instead.")
            return nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        losses = torch.zeros(len(self.names), device=preds.device)
        aggr_loss = torch.zeros(1, device=preds.device)

        losses = torch.stack([w * fn(preds, targets) for w, fn in zip(self.weights, self.losses)])
        aggr_loss = losses.sum()
        return aggr_loss, losses


# --- Example Test ---
if __name__ == "__main__":
    from landslide.dtypes import IterableSimpleNamespace as NDict

    test_cases = [
        "binary_cross_entropy",
        "weighted_binary_cross_entropy",
        "cross_entropy",  # unknown fallback case
        "focal_loss",
        "lovasz_loss",
        "dice_loss",
        "binary_cross_entropy+lovasz_loss",
        "binary_cross_entropy+dice_loss",
        "focal_loss+lovasz_loss",
        "focal_loss+dice_loss",
        "[1.5]focal_loss+[1.0]lovasz_loss",
    ]

    for criterion_str in test_cases:
        print(f"Criterion: {criterion_str}")
        model = None
        hyp = NDict(criterion=criterion_str)
        data = {"nc": 1, "pos_weights": [1.0]}
        device = "cpu"
        loss_fn = AutoCriterion(criterion_str, model, hyp, data, device)
        print(f"Parsed loss: {loss_fn}\n")

        preds = torch.randn(2, 1, 4, 4, requires_grad=True, device=device).float()
        targets = torch.randint(0, 2, (2, 1, 4, 4), device=device).float()
        loss, losses = loss_fn(preds, targets)
        print(f"Loss: {loss.item()}")
        print(f"Losses: {losses}\n")
