import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Standalone Dice loss.

    NOTE:
    This class is kept for experiments/baselines only.

    The native Mask2Former supervised training DOES NOT use
    this class directly. Its Dice loss is calculated inside
    Mask2Former's native SetCriterion after Hungarian matching.
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1).float()

        intersection = (probs_flat * targets_flat).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs_flat.sum()
            + targets_flat.sum()
            + self.smooth
        )

        return 1.0 - dice


class BinaryBCEDiceLoss(nn.Module):
    """
    Conventional binary segmentation baseline:

        L = BCE + Dice

    IMPORTANT:
    This is NOT the native Mask2Former supervised loss.

    Use this only if you intentionally want a conventional
    semantic-segmentation baseline such as U-Net.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, probs, targets):
        probs = torch.clamp(
            probs,
            min=1e-7,
            max=1.0 - 1e-7
        )

        targets = targets.float()

        loss_bce = self.bce(probs, targets)
        loss_dice = self.dice(probs, targets)

        return (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
        )


class UnsupervisedConsistencyLoss(nn.Module):
    """
    Mean Teacher consistency loss.

    This is NOT part of the supervised Mask2Former loss.
    It will be used later for L_unsup.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_probs, teacher_probs):

        student_probs = torch.clamp(
            student_probs,
            min=1e-7,
            max=1.0 - 1e-7
        )

        teacher_probs = torch.clamp(
            teacher_probs,
            min=1e-7,
            max=1.0 - 1e-7
        )

        return self.mse(
            student_probs,
            teacher_probs
        )


class SAMGuidedSoftLoss(nn.Module):
    """
    SAM-guided loss.

    This is NOT part of the supervised Mask2Former loss.
    It will be used later for L_sam.
    """

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(
            reduction="none"
        )

    def forward(
        self,
        preds,
        targets,
        uncertainties
    ):

        preds = preds.float()
        targets = targets.float()
        uncertainties = uncertainties.float()

        pixel_loss = self.mse(
            preds,
            targets
        )

        soft_weights = torch.exp(
            -uncertainties
        )

        weighted_loss = (
            pixel_loss * soft_weights
        )

        return weighted_loss.sum() / (
            soft_weights.sum() + 1e-8
        )


class Mask2FormerSupervisedLoss:
    """
    Wrapper for the NATIVE Mask2Former supervised objective.

    The actual loss computation is performed by Mask2Former's
    native SetCriterion.

    It includes:

        1. Hungarian matching
        2. Classification CE
        3. Mask BCE
        4. Dice loss
        5. Auxiliary decoder losses

    Therefore we MUST NOT manually calculate:

        BCE + Dice

    outside Mask2Former.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, loss_dict):
        """
        loss_dict is returned directly by:

            student_model(batch)

        Each value corresponds to one native Mask2Former
        loss component, e.g.:

            loss_ce
            loss_mask
            loss_dice
            loss_ce_0
            loss_mask_0
            loss_dice_0
            ...

        The native Mask2Former model already applies the
        configured loss weights.
        """

        return sum(
            loss
            for loss in loss_dict.values()
            if torch.is_tensor(loss)
        )
