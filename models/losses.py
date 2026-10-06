import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1).float()
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class SupervisedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, probs, targets):
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        loss_bce = self.bce(probs, targets.float())
        loss_dice = self.dice(probs, targets)
        return loss_bce + loss_dice

class UnsupervisedConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_probs, teacher_probs):
        student_probs = torch.clamp(student_probs, min=1e-7, max=1.0 - 1e-7)
        teacher_probs = torch.clamp(teacher_probs, min=1e-7, max=1.0 - 1e-7)
        return self.mse(student_probs, teacher_probs)

class SAMGuidedSoftLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, preds, targets, uncertainties):
        preds = preds.float()
        targets = targets.float()
        uncertainties = uncertainties.float()
        pixel_loss = self.mse(preds, targets)
        soft_weights = torch.exp(-uncertainties)
        weighted_loss = pixel_loss * soft_weights
        return weighted_loss.sum() / (soft_weights.sum() + 1e-8)
