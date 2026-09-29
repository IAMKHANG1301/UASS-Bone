import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        # probs đã nằm trong khoảng [0, 1]
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float() # Ép về float để nhân với probs an toàn tuyệt đối
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class SupervisedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss() # Thay vì BCEWithLogitsLoss vì Mask2Former đã trả về xác suất
        self.dice = DiceLoss()

    def forward(self, probs, targets):
        loss_bce = self.bce(probs, targets.float())
        loss_dice = self.dice(probs, targets)
        return loss_bce + loss_dice

class UnsupervisedConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_probs, teacher_probs):
        # Cả 2 đều đã là xác suất [0, 1]
        return self.mse(student_probs, teacher_probs)
