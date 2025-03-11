import torch

def _binary_confusion_matrix_update(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the bins to update the confusion matrix with."""
    unique_mapping = (target * 2 + preds).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=4)
    return bins.reshape(2, 2)


class BinaryConfusionMatrix:
    def __init__(self, threshold: float = 0.5):
        self.confmat = torch.zeros(2, 2)
        self.threshold = threshold
    
    def update(self, preds: torch.Tensor, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        if preds.is_floating_point():
            preds = (torch.sigmoid(preds) > self.threshold).long()
        self.confmat += _binary_confusion_matrix_update(preds, targets)

    def compute(self):
        return self.confmat.flatten()
        

