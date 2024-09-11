import torch
from torch import nn


class F1Loss(nn.Module):
    def __init__(
            self,
            return_cross_entropy: bool = False,
            epsilon: float = 1e-7
    ):
        """
        F1/CrossEntropy Loss module
        :param return_cross_entropy: Whether to return nn.CrossEntropyLoss()
        :param epsilon: Additional value for stable gradient computation
        """
        super(F1Loss, self).__init__()
        self.epsilon = epsilon
        self.ce = nn.CrossEntropyLoss() if return_cross_entropy else None

    def forward(self, y_pred, y_true):
        if self.ce is None:
            assert y_pred.ndim == 2
            assert y_true.ndim == 1
            y_true = torch.nn.functional.one_hot(y_true, 2).to(torch.float32)
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)

            tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
            tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
            fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
            fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)

            result = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
            return 1 - result.mean()
        else:
            return self.ce(y_pred, y_true)
