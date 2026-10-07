import os
import sys
import warnings
import random

# ============================================================
# Tắt warning không cần thiết từ Detectron2 / Mask2Former /
# PyTorch / Timm.
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


# ============================================================
# PATH CỦA CÁC REPOSITORY BÊN NGOÀI TRÊN KAGGLE
# ============================================================

sys.path.append(
    "/kaggle/working/efficientvit"
)

sys.path.append(
    "/kaggle/working/Mask2Former"
)


# ============================================================
# IMPORT
# ============================================================

import matplotlib
matplotlib.use("Agg")

import argparse

import numpy as np
import torch

import matplotlib.pyplot as plt

from torch.utils.data import (
    DataLoader,
    Subset,
    random_split
)

from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog
)

from data.dataset import (
    FracAtlasDataset,
    custom_collate
)

from models.specialist import (
    setup_config,
    build_specialist_model
)

from engine.trainer import (
    train_supervised
)


# ============================================================
# RANDOM SEED
#
# Đảm bảo:
#
#   train/validation split
#
# được giữ cố định giữa train và infer.
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# REGISTER DETECTRON2 METADATA
# ============================================================

def register_metadata():

    if "fracatlas_train" not in DatasetCatalog.list():

        DatasetCatalog.register(
            "fracatlas_train",
            lambda: []
        )

        MetadataCatalog.get(
            "fracatlas_train"
        ).set(
            thing_classes=["fracture"],
            stuff_classes=["background"],
            ignore_label=255
        )


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    image_dir,
    mask_dir
):

    dataset = FracAtlasDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None
    )

    return dataset


# ============================================================
# TRAIN / VALIDATION SPLIT
#
# IMPORTANT:
#
# Không dùng random.sample() riêng trong infer.
#
# Phải dùng đúng random_split + seed giống lúc train,
# nếu không validation set ở infer có thể khác validation
# set lúc train.
# ============================================================

def split_dataset(
    dataset,
    train_ratio,
    seed
):

    total_images = len(dataset)

    train_size = int(
        total_images * train_ratio
    )

    val_size = (
        total_images - train_size
    )

    generator = torch.Generator()

    generator.manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [
            train_size,
            val_size
        ],
        generator=generator
    )

    return (
        train_dataset,
        val_dataset
    )


# ============================================================
# BUILD DATALOADER
# ============================================================

def build_loader(
    dataset,
    batch_size,
    device,
    shuffle
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        collate_fn=custom_collate
    )


# ============================================================
# LOAD CHECKPOINT
#
# Hỗ trợ cả:
#
#   torch.save(model.state_dict(), path)
#
# và:
#
#   torch.save(
#       {
#           "model_state_dict": ...,
#           ...
#       },
#       path
#   )
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device
):

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"\n❌ Checkpoint không tồn tại:\n"
            f"{checkpoint_path}"
        )

    print()
    print("=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # --------------------------------------------------------
    # Case 1:
    #
    # checkpoint = model.state_dict()
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )

    if len(missing_keys) > 0:

        print(
            "\n⚠️ Missing keys:"
        )

        for key in missing_keys[:20]:

            print(
                f"    {key}"
            )

        if len(missing_keys) > 20:

            print(
                f"    ... "
                f"{len(missing_keys) - 20} more"
            )

    if len(unexpected_keys) > 0:

        print(
            "\n⚠️ Unexpected keys:"
        )

        for key in unexpected_keys[:20]:

            print(
                f"    {key}"
            )

        if len(unexpected_keys) > 20:

            print(
                f"    ... "
                f"{len(unexpected_keys) - 20} more"
            )

    print(
        "\n✅ Checkpoint loaded."
    )

    print("=" * 70)
    print()

    return checkpoint


# ============================================================
# EXTRACT GT MASK
#
# Dataset của bạn lưu:
#
#   instances.gt_masks
#
# dưới dạng:
#
#   [N, H, W]
#
# Với FracAtlas hiện tại:
#
#   N = 0 hoặc 1
#
# Nếu có nhiều instance thì merge thành semantic mask.
# ============================================================

def get_gt_mask(sample):

    instances = sample.get(
        "instances",
        None
    )

    if instances is None:

        return np.zeros(
            (
                sample["height"],
                sample["width"]
            ),
            dtype=bool
        )

    gt_masks = (
        instances.gt_masks
    )

    if isinstance(
        gt_masks,
        torch.Tensor
    ):

        gt_masks = (
            gt_masks.detach()
            .cpu()
            .numpy()
        )

    if gt_masks.shape[0] == 0:

        return np.zeros(
            (
                sample["height"],
                sample["width"]
            ),
            dtype=bool
        )

    # --------------------------------------------------------
    # Merge tất cả instance mask.
    #
    # Hiện tại dataset của bạn thường chỉ có 1 fracture mask.
    # --------------------------------------------------------

    gt_mask = np.any(
        gt_masks,
        axis=0
    )

    return gt_mask.astype(bool)


# ============================================================
# GET PREDICTION MASK
#
# Mask2Former semantic inference trả:
#
#   outputs[0]["sem_seg"]
#
# shape:
#
#   [num_classes, H, W]
#
# Dataset hiện tại:
#
#   NUM_CLASSES = 1
#
# nên lấy:
#
#   sem_seg[0]
#
# rồi sigmoid + threshold.
# ============================================================

def get_prediction(
    model,
    sample,
    device,
    threshold=0.5,
    query_threshold=0.5
):

    from detectron2.structures import ImageList
    import torch.nn.functional as F

    image = sample["image"].to(device)

    height = sample["height"]
    width = sample["width"]

    # ----------------------------------------------------
    # MASK2FORMER PREPROCESSING
    # ----------------------------------------------------

    image = (
        image - model.pixel_mean
    ) / model.pixel_std

    images = ImageList.from_tensors(
        [image],
        model.size_divisibility
    )

    # ----------------------------------------------------
    # BACKBONE
    # ----------------------------------------------------

    features = model.backbone(
        images.tensor
    )

    # ----------------------------------------------------
    # MASK2FORMER HEAD
    # ----------------------------------------------------

    outputs = model.sem_seg_head(
        features
    )

    pred_logits = outputs[
        "pred_logits"
    ][0]

    pred_masks = outputs[
        "pred_masks"
    ][0]

    # ----------------------------------------------------
    # CLASS PROBABILITY
    #
    # [fracture, no-object]
    # ----------------------------------------------------

    class_probs = F.softmax(
        pred_logits,
        dim=-1
    )

    fracture_probs = class_probs[:, 0]

    # ----------------------------------------------------
    # BEST FRACTURE QUERY
    #
    # Dataset chỉ có tối đa 1 fracture instance/image
    # ----------------------------------------------------

    best_query = torch.argmax(
        fracture_probs
    )

    best_class_prob = fracture_probs[
        best_query
    ]

    # ----------------------------------------------------
    # NO FRACTURE
    # ----------------------------------------------------

    if best_class_prob < query_threshold:

        pred_prob = torch.zeros(
            (height, width),
            dtype=torch.float32,
            device=device
        )

        pred_mask = torch.zeros(
            (height, width),
            dtype=torch.bool,
            device=device
        )

        return (
            pred_prob.cpu().numpy(),
            pred_mask.cpu().numpy()
        )

    # ----------------------------------------------------
    # RAW MASK LOGITS
    # ----------------------------------------------------

    mask_logits = pred_masks[
        best_query
    ]

    # ----------------------------------------------------
    # SIGMOID
    #
    # Đây mới là sigmoid đúng chỗ.
    # pred_masks là raw mask logits.
    # ----------------------------------------------------

    mask_prob = torch.sigmoid(
        mask_logits
    )

    # ----------------------------------------------------
    # UPSAMPLE
    # ----------------------------------------------------

    mask_prob = F.interpolate(
        mask_prob[None, None],
        size=(
            images.tensor.shape[-2],
            images.tensor.shape[-1]
        ),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    # ----------------------------------------------------
    # REMOVE PADDING
    # ----------------------------------------------------

    mask_prob = mask_prob[
        :height,
        :width
    ]

    # ----------------------------------------------------
    # BINARY MASK
    # ----------------------------------------------------

    pred_mask = (
        mask_prob >= threshold
    )

    return (
        mask_prob.detach()
        .float()
        .cpu()
        .numpy(),

        pred_mask.detach()
        .cpu()
        .numpy()
    )


# ============================================================
# PIXEL-LEVEL METRICS
#
# TP:
#   GT = 1
#   Pred = 1
#
# FP:
#   GT = 0
#   Pred = 1
#
# FN:
#   GT = 1
#   Pred = 0
#
# TN:
#   GT = 0
#   Pred = 0
# ============================================================

def calculate_metrics(
    pred_mask,
    gt_mask
):

    pred = pred_mask.astype(bool)

    gt = gt_mask.astype(bool)

    tp = np.logical_and(
        pred,
        gt
    ).sum()

    fp = np.logical_and(
        pred,
        np.logical_not(gt)
    ).sum()

    fn = np.logical_and(
        np.logical_not(pred),
        gt
    ).sum()

    tn = np.logical_and(
        np.logical_not(pred),
        np.logical_not(gt)
    ).sum()

    # --------------------------------------------------------
    # Dice
    #
    # Dice =
    #
    #       2TP
    # ----------------
    #      2TP+FP+FN
    #
    # Nếu cả GT và prediction đều empty:
    # coi là Dice = 1.
    # --------------------------------------------------------

    dice_denominator = (
        2 * tp
        + fp
        + fn
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * tp
            /
            dice_denominator
        )

    # --------------------------------------------------------
    # IoU
    #
    #       TP
    # ----------------
    #      TP+FP+FN
    # --------------------------------------------------------

    iou_denominator = (
        tp
        + fp
        + fn
    )

    if iou_denominator == 0:

        iou = 1.0

    else:

        iou = (
            tp
            /
            iou_denominator
        )

    # --------------------------------------------------------
    # Precision
    #
    #       TP
    # ----------------
    #      TP + FP
    # --------------------------------------------------------

    precision_denominator = (
        tp + fp
    )

    if precision_denominator == 0:

        precision = 1.0 if (
            tp + fn == 0
        ) else 0.0

    else:

        precision = (
            tp
            /
            precision_denominator
        )

    # --------------------------------------------------------
    # Recall
    #
    #       TP
    # ----------------
    #      TP + FN
    # --------------------------------------------------------

    recall_denominator = (
        tp + fn
    )

    if recall_denominator == 0:

        recall = 1.0

    else:

        recall = (
            tp
            /
            recall_denominator
        )

    # --------------------------------------------------------
    # Accuracy
    #
    #       TP + TN
    # ----------------
    #    TP+TN+FP+FN
    # --------------------------------------------------------

    total = (
        tp
        + tn
        + fp
        + fn
    )

    if total == 0:

        accuracy = 1.0

    else:

        accuracy = (
            (tp + tn)
            /
            total
        )

    # --------------------------------------------------------
    # Foreground ratios
    # --------------------------------------------------------

    total_pixels = (
        gt.size
    )

    gt_fg_ratio = (
        gt.sum()
        /
        total_pixels
    )

    pred_fg_ratio = (
        pred.sum()
        /
        total_pixels
    )

    return {

        "dice":
            float(dice),

        "iou":
            float(iou),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "accuracy":
            float(accuracy),

        "tp":
            int(tp),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tn":
            int(tn),

        "gt_fg_ratio":
            float(gt_fg_ratio),

        "pred_fg_ratio":
            float(pred_fg_ratio)
    }


# ============================================================
# EVALUATE VALIDATION DICE
# ============================================================

def evaluate_validation_dice(
    model,
    dataset,
    device
):
    import torch
    
    with torch.no_grad():
        total_dice = 0.0
        
        for local_idx in range(len(dataset)):
            sample = dataset[local_idx]
            gt_mask = get_gt_mask(sample)
            
            _, pred_mask = get_prediction(
                model=model,
                sample=sample,
                device=device,
                threshold=0.5
            )
            
            metrics = calculate_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask
            )
            
            total_dice += metrics["dice"]
            
        mean_dice = total_dice / max(len(dataset), 1)
        return mean_dice


# ============================================================
# INFERENCE + DIAGNOSTICS
#
# Chạy trên VALIDATION SET.
#
# Output:
#
#   1. Average metrics
#   2. Best 4 images
#   3. Worst 4 images
#   4. Visualization
# ============================================================

def run_diagnostic_inference(
    model,
    dataset,
    device,
    threshold,
    output_filename
):

    print()
    print("=" * 70)
    print("MASK2FORMER INFERENCE + DIAGNOSTICS")
    print("=" * 70)

    print(
        f"\nDataset size: {len(dataset)}"
    )

    print(
        f"Prediction threshold: {threshold}"
    )

    model.eval()

    results = []

    # --------------------------------------------------------
    # Không cần gradient khi inference.
    # --------------------------------------------------------

    with torch.no_grad():

        for local_idx in range(
            len(dataset)
        ):

            sample = dataset[
                local_idx
            ]

            gt_mask = get_gt_mask(
                sample
            )

            pred_prob, pred_mask = (
                get_prediction(
                    model=model,
                    sample=sample,
                    device=device,
                    threshold=threshold
                )
            )

            metrics = calculate_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask
            )

            results.append({

                "dataset_idx":
                    local_idx,

                "image_id":
                    sample.get(
                        "image_id",
                        local_idx
                    ),

                "sample":
                    sample,

                "gt_mask":
                    gt_mask,

                "pred_prob":
                    pred_prob,

                "pred_mask":
                    pred_mask,

                "metrics":
                    metrics
            })

    if len(results) == 0:

        print(
            "\n❌ Validation dataset is empty."
        )

        return

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    metric_names = [

        "dice",

        "iou",

        "precision",

        "recall",

        "accuracy",

        "gt_fg_ratio",

        "pred_fg_ratio"
    ]

    print()
    print("=" * 70)
    print("OVERALL VALIDATION METRICS")
    print("=" * 70)

    for metric_name in metric_names:

        values = np.array([

            item["metrics"][
                metric_name
            ]

            for item in results

        ])

        print(
            f"{metric_name:18s}: "
            f"mean={values.mean():.6f} "
            f"| std={values.std():.6f} "
            f"| min={values.min():.6f} "
            f"| max={values.max():.6f}"
        )

    # ========================================================
    # GLOBAL CONFUSION MATRIX
    # ========================================================

    total_tp = sum(
        item["metrics"]["tp"]
        for item in results
    )

    total_fp = sum(
        item["metrics"]["fp"]
        for item in results
    )

    total_fn = sum(
        item["metrics"]["fn"]
        for item in results
    )

    total_tn = sum(
        item["metrics"]["tn"]
        for item in results
    )

    print()
    print("=" * 70)
    print("GLOBAL PIXEL CONFUSION MATRIX")
    print("=" * 70)

    print(
        f"TP = {total_tp}"
    )

    print(
        f"FP = {total_fp}"
    )

    print(
        f"FN = {total_fn}"
    )

    print(
        f"TN = {total_tn}"
    )

    # ========================================================
    # GLOBAL DICE / IOU
    #
    # Đây là metric tính từ tổng pixel của toàn bộ validation,
    # khác với mean Dice của từng ảnh.
    # ========================================================

    global_dice_den = (
        2 * total_tp
        + total_fp
        + total_fn
    )

    if global_dice_den == 0:

        global_dice = 1.0

    else:

        global_dice = (
            2.0 * total_tp
            /
            global_dice_den
        )

    global_iou_den = (
        total_tp
        + total_fp
        + total_fn
    )

    if global_iou_den == 0:

        global_iou = 1.0

    else:

        global_iou = (
            total_tp
            /
            global_iou_den
        )

    print(
        f"\nGlobal Dice : "
        f"{global_dice:.6f}"
    )

    print(
        f"Global IoU  : "
        f"{global_iou:.6f}"
    )

    # ========================================================
    # SORT BY DICE
    #
    # Best:
    #   Dice cao nhất
    #
    # Worst:
    #   Dice thấp nhất
    # ========================================================

    sorted_results = sorted(
        results,
        key=lambda x:
            x["metrics"]["dice"]
    )

    worst_results = (
        sorted_results[:4]
    )

    best_results = (
        sorted_results[-4:]
    )

    # ========================================================
    # PRINT BEST
    # ========================================================

    print()
    print("=" * 70)
    print("TOP 4 BEST IMAGES")
    print("=" * 70)

    for rank, item in enumerate(
        reversed(best_results),
        start=1
    ):

        m = item["metrics"]

        print(
            f"\n#{rank}"
        )

        print(
            f"    dataset_idx = "
            f"{item['dataset_idx']}"
        )

        print(
            f"    Dice       = "
            f"{m['dice']:.6f}"
        )

        print(
            f"    IoU        = "
            f"{m['iou']:.6f}"
        )

        print(
            f"    Precision  = "
            f"{m['precision']:.6f}"
        )

        print(
            f"    Recall     = "
            f"{m['recall']:.6f}"
        )

        print(
            f"    GT ratio   = "
            f"{m['gt_fg_ratio']:.6f}"
        )

        print(
            f"    Pred ratio = "
            f"{m['pred_fg_ratio']:.6f}"
        )

    # ========================================================
    # PRINT WORST
    # ========================================================

    print()
    print("=" * 70)
    print("BOTTOM 4 WORST IMAGES")
    print("=" * 70)

    for rank, item in enumerate(
        worst_results,
        start=1
    ):

        m = item["metrics"]

        print(
            f"\n#{rank}"
        )

        print(
            f"    dataset_idx = "
            f"{item['dataset_idx']}"
        )

        print(
            f"    Dice       = "
            f"{m['dice']:.6f}"
        )

        print(
            f"    IoU        = "
            f"{m['iou']:.6f}"
        )

        print(
            f"    Precision  = "
            f"{m['precision']:.6f}"
        )

        print(
            f"    Recall     = "
            f"{m['recall']:.6f}"
        )

        print(
            f"    GT ratio   = "
            f"{m['gt_fg_ratio']:.6f}"
        )

        print(
            f"    Pred ratio = "
            f"{m['pred_fg_ratio']:.6f}"
        )

    # ========================================================
    # VISUALIZATION
    #
    # 8 rows:
    #
    #   4 best
    #   4 worst
    #
    # 4 columns:
    #
    #   Original
    #   GT
    #   Prediction
    #   Overlay
    # ========================================================

    selected_results = (
        list(
            reversed(best_results)
        )
        +
        list(
            worst_results
        )
    )

    fig, axes = plt.subplots(
        8,
        4,
        figsize=(18, 32)
    )

    fig.suptitle(
        "Mask2Former Validation Diagnostics\n"
        "4 Best + 4 Worst Images",
        fontsize=20,
        fontweight="bold"
    )

    for row, item in enumerate(
        selected_results
    ):

        sample = item["sample"]

        gt_mask = item["gt_mask"]

        pred_mask = item["pred_mask"]

        metrics = item["metrics"]

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        image = (
            sample["image"]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # ----------------------------------------------------
        # Image normalization
        #
        # Dataset image is uint8-like [0,255].
        # ----------------------------------------------------

        if image.max() > 1.0:

            image_display = (
                image / 255.0
            )

        else:

            image_display = image

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        ax = axes[row, 0]

        ax.imshow(
            image_display
        )

        if row < 4:

            title_prefix = (
                f"BEST #{row + 1}"
            )

        else:

            title_prefix = (
                f"WORST #{row - 3}"
            )

        ax.set_title(
            f"{title_prefix}\n"
            f"ID={item['dataset_idx']}",
            fontsize=11
        )

        ax.axis("off")

        # ----------------------------------------------------
        # Ground Truth
        # ----------------------------------------------------

        ax = axes[row, 1]

        ax.imshow(
            gt_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            "Ground Truth",
            fontsize=11
        )

        ax.axis("off")

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        ax = axes[row, 2]

        ax.imshow(
            pred_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f"Prediction\n"
            f"Dice={metrics['dice']:.4f} "
            f"| IoU={metrics['iou']:.4f}",
            fontsize=11
        )

        ax.axis("off")

        # ----------------------------------------------------
        # Overlay
        #
        # Hiển thị:
        #
        #   Original image
        #   GT       = red
        #   Pred     = green
        #
        # Ý nghĩa:
        #
        #   Red only   = FN
        #   Green only = FP
        #   Yellow     = overlap
        # ----------------------------------------------------

        overlay = (
            image_display.copy()
        )

        if overlay.shape[-1] == 3:

            red = gt_mask
            green = pred_mask

            overlay[..., 0] = np.clip(
                overlay[..., 0]
                +
                0.45 * red,
                0,
                1
            )

            overlay[..., 1] = np.clip(
                overlay[..., 1]
                +
                0.45 * green,
                0,
                1
            )

        ax = axes[row, 3]

        ax.imshow(
            overlay
        )

        ax.set_title(
            "Overlay\n"
            "Red=GT | Green=Pred",
            fontsize=11
        )

        ax.axis("off")

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97
        ]
    )

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    plt.savefig(
        output_filename,
        dpi=150,
        bbox_inches="tight"
    )

    print()
    print(
        f"🖼️ Diagnostic figure saved to:"
    )

    print(
        f"   {output_filename}"
    )

    plt.close(fig)

    # ========================================================
    # DIAGNOSTIC INTERPRETATION
    # ========================================================

    mean_dice = np.mean([

        x["metrics"]["dice"]

        for x in results
    ])

    mean_iou = np.mean([

        x["metrics"]["iou"]

        for x in results
    ])

    mean_pred_ratio = np.mean([

        x["metrics"]["pred_fg_ratio"]

        for x in results
    ])

    mean_gt_ratio = np.mean([

        x["metrics"]["gt_fg_ratio"]

        for x in results
    ])

    print()
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"\nMean Dice:"
        f" {mean_dice:.6f}"
    )

    print(
        f"Mean IoU:"
        f" {mean_iou:.6f}"
    )

    print(
        f"Mean GT foreground ratio:"
        f" {mean_gt_ratio:.6f}"
    )

    print(
        f"Mean predicted foreground ratio:"
        f" {mean_pred_ratio:.6f}"
    )

    # --------------------------------------------------------
    # Cảnh báo prediction collapse
    # --------------------------------------------------------

    if mean_pred_ratio < 0.001:

        print(
            "\n⚠️ WARNING:"
        )

        print(
            "Prediction foreground ratio rất thấp."
        )

        print(
            "Model có thể đang dự đoán gần như toàn background."
        )

    elif (
        mean_pred_ratio
        >
        max(
            mean_gt_ratio * 3.0,
            0.05
        )
    ):

        print(
            "\n⚠️ WARNING:"
        )

        print(
            "Prediction foreground ratio lớn hơn GT rất nhiều."
        )

        print(
            "Có khả năng model đang over-segment."
        )

    else:

        print(
            "\n✅ Foreground ratio không có dấu hiệu "
            "collapse rõ ràng."
        )

    # --------------------------------------------------------
    # Cảnh báo precision / recall
    # --------------------------------------------------------

    mean_precision = np.mean([

        x["metrics"]["precision"]

        for x in results
    ])

    mean_recall = np.mean([

        x["metrics"]["recall"]

        for x in results
    ])

    if (
        mean_precision
        >
        mean_recall * 1.5
    ):

        print(
            "\n⚠️ Recall thấp tương đối."
        )

        print(
            "Model có xu hướng bỏ sót fracture."
        )

    if (
        mean_recall
        >
        mean_precision * 1.5
    ):

        print(
            "\n⚠️ Precision thấp tương đối."
        )

        print(
            "Model có xu hướng dự đoán fracture quá nhiều."
        )

    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=
        "UASS-Bone "
        "EfficientViT + Mask2Former "
        "Train / Fine-tuning / Inference"
    )


    # ========================================================
    # MODE
    # ========================================================

    parser.add_argument(

        "--mode",

        type=str,

        required=True,

        choices=[
            "train",
            "fine-tuning",
            "infer"
        ],

        help=
        "train | fine-tuning | infer"
    )


    # ========================================================
    # DATASET
    # ========================================================

    parser.add_argument(

        "--image_dir",

        type=str,

        required=True
    )

    parser.add_argument(

        "--mask_dir",

        type=str,

        required=True
    )


    # ========================================================
    # TRAINING
    # ========================================================

    parser.add_argument(

        "--epochs",

        type=int,

        default=100
    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=2
    )


    # ========================================================
    # LEARNING RATE
    #
    # Train:
    #
    #   head     = 5e-5
    #   backbone = 5e-6
    #
    # Fine-tuning:
    #
    #   head     = 1e-5
    #   backbone = 1e-6
    #
    # Fine-tuning LR thấp hơn để tránh phá hủy
    # representation đã học.
    # ========================================================

    parser.add_argument(

        "--lr",

        type=float,

        default=1e-4
    )

    parser.add_argument(

        "--backbone_lr",

        type=float,

        default=1e-5
    )


    # ========================================================
    # FINE-TUNING LR
    #
    # Nếu không truyền:
    #
    #   lr = 1e-5
    #   backbone_lr = 1e-6
    # ========================================================

    parser.add_argument(

        "--finetune_lr",

        type=float,

        default=1e-5
    )

    parser.add_argument(

        "--finetune_backbone_lr",

        type=float,

        default=1e-6
    )


    # ========================================================
    # DEVICE
    # ========================================================

    parser.add_argument(

        "--device",

        type=str,

        default=
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    parser.add_argument(

        "--train_ratio",

        type=float,

        default=0.8
    )

    parser.add_argument(

        "--seed",

        type=int,

        default=42
    )


    # ========================================================
    # OVERFIT DEBUG
    # ========================================================

    parser.add_argument(

        "--overfit",

        action="store_true",

        help=
        "Train only a small subset "
        "to verify supervised convergence."
    )

    parser.add_argument(

        "--overfit_images",

        type=int,

        default=8
    )


    # ========================================================
    # CHECKPOINT
    #
    # train:
    #
    #   --save_path specialist_supervised.pth
    #
    # fine-tuning / infer:
    #
    #   --checkpoint specialist_supervised.pth
    # ========================================================

    parser.add_argument(

        "--checkpoint",

        type=str,

        default=None
    )

    parser.add_argument(

        "--save_path",

        type=str,

        default=
        "specialist_supervised.pth"
    )


    # ========================================================
    # INFERENCE SETTINGS
    # ========================================================

    parser.add_argument(

        "--threshold",

        type=float,

        default=0.5,

        help=
        "Binary threshold for predicted semantic mask."
    )

    parser.add_argument(
        "--infer_on_train",
        action="store_true",
        help="[DEBUG] Run inference on the 80% TRAIN SET instead of validation."
    )

    parser.add_argument(

        "--output",

        type=str,

        default=
        "mask2former_diagnostics.png"
    )


    args = parser.parse_args()


    # ========================================================
    # VALIDATE ARGUMENTS
    # ========================================================

    if args.mode in [
        "fine-tuning",
        "infer"
    ]:

        if args.checkpoint is None:

            raise ValueError(

                "\n❌ --checkpoint là bắt buộc "
                f"khi --mode {args.mode}\n\n"

                "Ví dụ:\n"

                "--mode fine-tuning "
                "--checkpoint specialist_supervised.pth\n"

                "hoặc:\n"

                "--mode infer "
                "--checkpoint specialist_supervised.pth"
            )


    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        args.seed
    )


    # ========================================================
    # DEVICE
    # ========================================================

    print()
    print(
        f"🖥️ Device: {args.device}"
    )


    # ========================================================
    # DETECTRON2 METADATA
    # ========================================================

    register_metadata()


    # ========================================================
    # BUILD CONFIG
    # ========================================================

    print()
    print(
        "🔧 Building Mask2Former configuration..."
    )

    cfg = setup_config()


    # ========================================================
    # BUILD MODEL
    #
    # Cả 3 mode đều cần cùng architecture.
    #
    # Điều này cực kỳ quan trọng:
    #
    # checkpoint phải được load vào đúng architecture/config
    # với lúc training.
    # ========================================================

    print(
        "🧠 Building EfficientViT-B0 + Mask2Former..."
    )

    model = build_specialist_model(
        cfg
    )

    model.to(
        args.device
    )


    # ========================================================
    # DATASET
    # ========================================================

    print()
    print(
        f"📂 Image directory:"
        f"\n   {args.image_dir}"
    )

    print(
        f"📂 Mask directory:"
        f"\n   {args.mask_dir}"
    )

    dataset = build_dataset(

        image_dir=args.image_dir,

        mask_dir=args.mask_dir
    )

    total_images = len(
        dataset
    )

    print()
    print(
        f"📊 Total images: "
        f"{total_images}"
    )


    # ========================================================
    # SPLIT
    #
    # Dùng cho cả train và inference.
    #
    # Vì seed giống nhau nên:
    #
    # train mode:
    #   train_dataset = 80%
    #   val_dataset   = 20%
    #
    # infer mode:
    #   lấy đúng validation 20% đó.
    # ========================================================

    train_dataset, val_dataset = (
        split_dataset(

            dataset=dataset,

            train_ratio=
                args.train_ratio,

            seed=args.seed
        )
    )


    print()
    print(
        "📊 Dataset split:"
    )

    print(
        f"    Total      : "
        f"{total_images}"
    )

    print(
        f"    Train      : "
        f"{len(train_dataset)}"
    )

    print(
        f"    Validation : "
        f"{len(val_dataset)}"
    )


    # ========================================================
    # ========================================================
    # MODE 1: TRAIN
    # ========================================================
    # ========================================================

    if args.mode == "train":

        print()
        print("=" * 70)
        print("MODE: TRAIN")
        print("=" * 70)


        # ----------------------------------------------------
        # OVERFIT MODE
        # ----------------------------------------------------

        if args.overfit:

            n = min(

                args.overfit_images,

                len(train_dataset)
            )

            train_dataset = Subset(

                train_dataset,

                list(
                    range(n)
                )
            )

            print()
            print(
                "⚠️ OVERFIT DEBUG MODE"
            )

            print(
                f"Training images: "
                f"{n}"
            )

            print(
                "Teacher: OFF"
            )

            print(
                "EMA: OFF"
            )

            print(
                "L_unsup: OFF"
            )

            print(
                "L_sam: OFF"
            )


        # ----------------------------------------------------
        # DATALOADER
        # ----------------------------------------------------

        train_loader = build_loader(

            dataset=train_dataset,

            batch_size=args.batch_size,

            device=args.device,

            shuffle=True
        )


        # ----------------------------------------------------
        # PARAMETER GROUPS
        #
        # Backbone LR nhỏ hơn head LR.
        # ----------------------------------------------------

        backbone_params = []

        head_params = []


        for name, param in (
            model.named_parameters()
        ):

            if not param.requires_grad:

                continue


            if name.startswith(
                "backbone."
            ):

                backbone_params.append(
                    param
                )

            else:

                head_params.append(
                    param
                )


        print()
        print(
            "🔧 Optimizer:"
        )

        print(
            f"    Backbone LR: "
            f"{args.backbone_lr}"
        )

        print(
            f"    Head LR    : "
            f"{args.lr}"
        )


        optimizer = torch.optim.AdamW(

            [

                {
                    "params":
                        backbone_params,

                    "lr":
                        args.backbone_lr
                },

                {
                    "params":
                        head_params,

                    "lr":
                        args.lr
                }

            ],

            weight_decay=0.05
        )


        # ----------------------------------------------------
        # CHECK DATASET
        # ----------------------------------------------------

        first_batch = next(
            iter(train_loader)
        )

        print()
        print(
            "🔍 Dataset sanity check:"
        )

        print(
            f"    Batch size: "
            f"{len(first_batch)}"
        )

        print(
            f"    Image shape: "
            f"{first_batch[0]['image'].shape}"
        )

        print(
            f"    GT masks shape: "
            f"{first_batch[0]['instances'].gt_masks.shape}"
        )

        print(
            f"    GT classes: "
            f"{first_batch[0]['instances'].gt_classes}"
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print()
        print(
            "🚀 Starting supervised training..."
        )

        train_supervised(

            student_model=model,

            labeled_loader=train_loader,

            optimizer=optimizer,

            epochs=args.epochs,

            device=args.device,

            max_grad_norm=1.0,
            
            val_dataset=val_dataset,
            
            eval_fn=evaluate_validation_dice,
            
            save_path=args.save_path
        )

        print()
        print(
            "✅ TRAIN COMPLETE"
        )

        print()
        print(
            "📌 Next recommended step:"
        )

        print(
            "Run --mode infer "
            "to inspect validation Dice/IoU "
            "and the 4 best + 4 worst images."
        )


    # ========================================================
    # ========================================================
    # MODE 2: FINE-TUNING
    # ========================================================
    # ========================================================

    elif args.mode == "fine-tuning":

        print()
        print("=" * 70)
        print("MODE: FINE-TUNING")
        print("=" * 70)


        # ----------------------------------------------------
        # LOAD PREVIOUS MODEL
        #
        # Không train từ random initialization.
        # ----------------------------------------------------

        load_checkpoint(

            model=model,

            checkpoint_path=
                args.checkpoint,

            device=args.device
        )


        # ----------------------------------------------------
        # Fine-tuning vẫn dùng TRAIN SET.
        #
        # Validation vẫn giữ nguyên.
        # ----------------------------------------------------

        train_loader = build_loader(

            dataset=train_dataset,

            batch_size=args.batch_size,

            device=args.device,

            shuffle=True
        )


        # ----------------------------------------------------
        # PARAMETER GROUPS
        # ----------------------------------------------------

        backbone_params = []

        head_params = []


        for name, param in (
            model.named_parameters()
        ):

            if not param.requires_grad:

                continue


            if name.startswith(
                "backbone."
            ):

                backbone_params.append(
                    param
                )

            else:

                head_params.append(
                    param
                )


        print()
        print(
            "🔧 Fine-tuning optimizer:"
        )

        print(
            f"    Backbone LR: "
            f"{args.finetune_backbone_lr}"
        )

        print(
            f"    Head LR    : "
            f"{args.finetune_lr}"
        )


        # ----------------------------------------------------
        # Fine-tuning dùng LR nhỏ hơn train mới.
        #
        # Mục đích:
        #
        #   không phá hủy representation đã học.
        # ----------------------------------------------------

        optimizer = torch.optim.AdamW(

            [

                {
                    "params":
                        backbone_params,

                    "lr":
                        args.finetune_backbone_lr
                },

                {
                    "params":
                        head_params,

                    "lr":
                        args.finetune_lr
                }

            ],

            weight_decay=0.05
        )


        # ----------------------------------------------------
        # CONTINUE TRAINING
        # ----------------------------------------------------

        print()
        print(
            "🚀 Starting fine-tuning..."
        )

        train_supervised(

            student_model=model,

            labeled_loader=train_loader,

            optimizer=optimizer,

            epochs=args.epochs,

            device=args.device,

            max_grad_norm=1.0,
            
            val_dataset=val_dataset,
            
            eval_fn=evaluate_validation_dice,
            
            save_path=args.save_path
        )


        print()
        print(
            "✅ FINE-TUNING COMPLETE"
        )


    # ========================================================
    # ========================================================
    # MODE 3: INFERENCE
    # ========================================================
    # ========================================================

    elif args.mode == "infer":

        print()
        print("=" * 70)
        print("MODE: INFERENCE + MODEL DIAGNOSTICS")
        print("=" * 70)


        # ----------------------------------------------------
        # LOAD TRAINED MODEL
        # ----------------------------------------------------

        load_checkpoint(

            model=model,

            checkpoint_path=
                args.checkpoint,

            device=args.device
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Inference trên validation 20%.
        #
        # Không đánh giá trên train set để tránh đánh giá
        # model bằng dữ liệu mà nó đã nhìn thấy.
        #
        # [DEBUG OVERRIDE]: Nếu --infer_on_train được bật,
        # sẽ chạy đánh giá trên chính tập Train 80%.
        # ----------------------------------------------------

        target_dataset = train_dataset if args.infer_on_train else val_dataset

        if args.infer_on_train:
            print("\n⚠️ [DEBUG] RUNNING INFERENCE ON 80% TRAIN SET INSTEAD OF VALIDATION!\n")

        run_diagnostic_inference(

            model=model,

            dataset=target_dataset,

            device=args.device,

            threshold=args.threshold,

            output_filename=args.output
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
