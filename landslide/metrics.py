from typing import Literal, Optional, Self, Tuple

import torch
from torch import Tensor


def normalize_logits_if_needed(
    tensor: Tensor, normalization: Literal["sigmoid", "softmax"]
) -> Tensor:
    """Normalize logits if needed.

    If input tensor is outside the [0,1] we assume that logits are provided and apply the normalization.
    Use torch.where to prevent device-host sync.

    Args:
        tensor: input tensor that may be logits or probabilities
        normalization: normalization method, either 'sigmoid' or 'softmax'

    Returns:
        normalized tensor if needed

    Example:
        >>> import torch
        >>> tensor = torch.tensor([-1.0, 0.0, 1.0])
        >>> normalize_logits_if_needed(tensor, normalization="sigmoid")
        tensor([0.2689, 0.5000, 0.7311])
        >>> tensor = torch.tensor([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
        >>> normalize_logits_if_needed(tensor, normalization="softmax")
        tensor([[0.0900, 0.2447, 0.6652],
                [0.6652, 0.2447, 0.0900]])
        >>> tensor = torch.tensor([0.0, 0.5, 1.0])
        >>> normalize_logits_if_needed(tensor, normalization="sigmoid")
        tensor([0.0000, 0.5000, 1.0000])

    """
    # decrease sigmoid on cpu .
    if tensor.device == torch.device("cpu"):
        if not torch.all((tensor >= 0) * (tensor <= 1)):
            tensor = (
                tensor.sigmoid()
                if normalization == "sigmoid"
                else torch.softmax(tensor, dim=1)
            )
        return tensor

    # decrease device-host sync on device .
    condition = ((tensor < 0) | (tensor > 1)).any()
    return torch.where(
        condition,
        torch.sigmoid(tensor)
        if normalization == "sigmoid"
        else torch.softmax(tensor, dim=1),
        tensor,
    )


def _binary_confusion_matrix_format(
    preds: Tensor, target: Tensor, threshold: float, ignore_index: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Check inputs and format for binary confusion matrix computation."""
    preds = preds.flatten()
    target = target.flatten()

    if preds.is_floating_point():
        preds = normalize_logits_if_needed(preds, normalization="sigmoid")
        preds = preds > threshold
    return preds, target


def _binary_confusion_matrix_update(preds: Tensor, target: Tensor) -> Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = (target * 2 + preds).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=4)
    return bins.reshape(2, 2)


def _confusion_matrix_reduce(
    confmat: Tensor, normalize: Optional[Literal["true", "pred", "all", "none"]] = None
) -> Tensor:
    """Reduce an un-normalized confusion matrix.

    Args:
        confmat: un-normalized confusion matrix
        normalize: normalization method.
            - `"true"` will divide by the sum of the column dimension.
            - `"pred"` will divide by the sum of the row dimension.
            - `"all"` will divide by the sum of the full matrix
            - `"none"` or `None` will apply no reduction.

    Returns:
        Normalized confusion matrix

    """
    allowed_normalize = ("true", "pred", "all", "none", None)
    if normalize not in allowed_normalize:
        raise ValueError(
            f"Argument `normalize` needs to one of the following: {allowed_normalize}"
        )
    if normalize is not None and normalize != "none":
        confmat = confmat.float() if not confmat.is_floating_point() else confmat
        if normalize == "true":
            confmat = confmat / confmat.sum(dim=-1, keepdim=True)
        elif normalize == "pred":
            confmat = confmat / confmat.sum(dim=-2, keepdim=True)
        elif normalize == "all":
            confmat = confmat / confmat.sum(dim=[-2, -1], keepdim=True)

        nan_elements = confmat[torch.isnan(confmat)].nelement()
        if nan_elements:
            confmat[torch.isnan(confmat)] = 0
            # TODO: ON RANK 0 warn
            print(
                f"{nan_elements} NaN values found in confusion matrix have been replaced with zeros."
            )
    return confmat


def _binary_confusion_matrix_compute(
    confmat: Tensor, normalize: bool = False
) -> Tensor:
    """Compute the binary confusion matrix."""
    if normalize:
        confmat = _confusion_matrix_reduce(confmat, normalize="all")
    return confmat


class BinaryConfusionMatrix:
    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        normalize: bool = False,
    ) -> None:
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.confmat = torch.zeros(2, 2, dtype=torch.long)

    def to(self: Self, device: torch.device) -> Self:
        """Move the confusion matrix to a device."""
        self.confmat = self.confmat.to(device)
        return self

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _binary_confusion_matrix_format(
            preds, target, self.threshold, self.ignore_index
        )
        confmat = _binary_confusion_matrix_update(preds, target)
        self.confmat += confmat
        return confmat
    
    def __call__(self, preds: Tensor, target: Tensor):
        self.update(preds, target)
        return self.confmat

    def compute(self, normalize: bool = False, flatten: bool = False) -> Tensor:
        """Compute confusion matrix."""
        mat = _binary_confusion_matrix_compute(
            self.confmat, self.normalize or normalize
        )
        if flatten:
            mat = mat.flatten()
        return mat

    @property
    def tp(self):
        return self.confmat[1, 1].item()

    @property
    def fp(self):
        return self.confmat[0, 1].item()

    @property
    def fn(self):
        return self.confmat[1, 0].item()

    @property
    def tn(self):
        return self.confmat[0, 0].item()

    @property
    def precision(self):
        tp = self.tp
        fp = self.fp
        zero_div = tp + fp == 0
        return tp / (tp + fp) if not zero_div else 0.0

    @property
    def recall(self):
        tp = self.tp
        fn = self.fn
        zero_div = tp + fn == 0
        return tp / (tp + fn) if not zero_div else 0.0

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-7)
    
    @property
    def f1(self):
        precision = self.precision
        recall = self.recall
        zero_div = precision + recall == 0
        if zero_div:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def metrics(self, prefix: Optional[str] = ""):
        return {
            f"{prefix}True Positives": self.tp,
            f"{prefix}False Positives": self.fp,
            f"{prefix}False Negatives": self.fn,
            f"{prefix}True Negatives": self.tn,
            f"{prefix}Precision": self.precision,
            f"{prefix}Recall": self.recall,
            f"{prefix}Accuracy": self.accuracy,
            f"{prefix}F1-Score": self.f1,
        }
