import torch
import torch.nn as nn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Trọng số phạt dự đoán sai nền thành nứt (FP)
        self.beta = beta    # Trọng số phạt bỏ sót vết nứt (FN)
        self.gamma = gamma  # Hệ số Focal triệt tiêu vùng nền
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.float().view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - Tversky) ** self.gamma

class SupervisedLoss(nn.Module):
    def __init__(self, loss_type="bce_dice"):
        super().__init__()
        self.loss_type = loss_type
        self.bce = nn.BCEWithLogitsLoss() 
        self.tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0)
        
    def forward(self, preds, targets):
        if self.loss_type == "tversky":
            return self.tversky(preds, targets)
        else:
            loss_bce = self.bce(preds, targets.float())
            preds_sig = torch.sigmoid(preds)
            intersection = (preds_sig * targets).sum()
            dice_loss = 1 - (2. * intersection + 1e-6) / (preds_sig.sum() + targets.sum() + 1e-6)
            return loss_bce + dice_loss

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
        # Không dùng reduction='mean' để tự tính trung bình theo trọng số
        self.mse = nn.MSELoss(reduction='none') 
        
    def forward(self, preds, targets, uncertainties):
        """
        preds, targets, uncertainties: [B, 1, H, W], dtype torch.float32
        """
        # Ép kiểu an toàn
        preds = preds.float()
        targets = targets.float()
        uncertainties = uncertainties.float()
        
        # 1. Tính sai số bình phương từng pixel
        pixel_loss = self.mse(preds, targets) # [B, 1, H, W]
        
        # 2. Cơ chế Soft Masking: Phân phối trọng số liên tục
        soft_weights = torch.exp(-uncertainties) # [B, 1, H, W]
        
        # 3. Tính toán mất mát có chọn lọc vùng
        weighted_loss = pixel_loss * soft_weights # [B, 1, H, W]
        
        # 4. Trả về giá trị vô hướng (Scalar Tensor)
        return weighted_loss.sum() / (soft_weights.sum() + 1e-8)
