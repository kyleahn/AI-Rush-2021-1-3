import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target, weight=None):
        if len(target.shape) == 1:
            target = target.unsqueeze(-1)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            onehot = torch.zeros(pred.shape, device=pred.device)
            onehot += self.smoothing / (pred.shape[-1] - 1)
            mask = (target != -100).float()
            target = torch.where(target == -100, torch.zeros(target.shape, device=target.device, dtype=torch.long), target)
            onehot = torch.scatter(onehot, -1, target, 1 - self.smoothing)
        onehot = torch.sum(-onehot * pred, dim=-1)
        onehot = onehot * mask.reshape(-1)
        if weight is not None:
            onehot = onehot * weight.reshape(-1)
        return torch.mean(onehot)
