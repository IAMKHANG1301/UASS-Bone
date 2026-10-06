import os
import sys
import warnings

# Tắt các cảnh báo FutureWarning và UserWarning rác từ Mask2Former / PyTorch / Timm
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append("/kaggle/working/efficientvit")
sys.path.append("/kaggle/working/Mask2Former")

import argparse
import torch
from torch.utils.data import DataLoader, Subset, random_split
from detectron2.data import DatasetCatalog, MetadataCatalog

from data.dataset import FracAtlasDataset, custom_collate
from models.specialist import setup_config, build_specialist_model
from engine.trainer import train_supervised


def main():

    parser = argparse.ArgumentParser(
        description="UASS-Bone supervised Mask2Former training"
    )

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

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5
    )

    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=5e-6
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ============================================================
    # 80/20 TRAIN / VALIDATION SPLIT
    # ============================================================
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of dataset used for supervised training."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/validation split."
    )

    # ============================================================
    # OVERFIT DEBUG MODE
    #
    # Chỉ dùng để kiểm tra model có thể overfit một số ảnh nhỏ
    # hay không.
    #
    # KHÔNG dùng --overfit khi train chính thức 80% dataset.
    # ============================================================
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit a very small labeled subset to verify supervised convergence."
    )

    parser.add_argument(
        "--overfit_images",
        type=int,
        default=8
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="specialist_supervised.pth"
    )

    args = parser.parse_args()

    # ============================================================
    # Detectron2 metadata
    # ============================================================

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
    # BUILD MASK2FORMER CONFIGURATION
    # ============================================================

    print("\n🔧 Building Mask2Former configuration...")

    cfg = setup_config()

    # ============================================================
    # BUILD EFFICIENTVIT-B0 + MASK2FORMER
    # ============================================================

    print(
        "🧠 Building EfficientViT-B0 + Mask2Former..."
    )

    model = build_specialist_model(cfg)

    model.to(args.device)

    # ============================================================
    # DATASET
    # ============================================================

    print(
        f"\n📂 Image directory:\n{args.image_dir}"
    )

    print(
        f"📂 Mask directory:\n{args.mask_dir}"
    )

    dataset = FracAtlasDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        transform=None
    )

    total_images = len(dataset)

    print(
        f"\n📊 Total images: {total_images}"
    )

    # ============================================================
    # 80% TRAIN / 20% VALIDATION SPLIT
    #
    # Với dataset hiện tại:
    #
    # Total = 717
    #
    # Train = floor(717 * 0.8) = 573
    # Val   = 717 - 573       = 144
    #
    # Generator + seed đảm bảo lần chạy sau vẫn có
    # đúng train/validation split này.
    # ============================================================

    train_size = int(
        total_images * args.train_ratio
    )

    val_size = (
        total_images - train_size
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print("\n📊 Dataset split:")

    print(
        f"    Total      : {total_images}"
    )

    print(
        f"    Train      : {len(train_dataset)} "
        f"({args.train_ratio * 100:.1f}%)"
    )

    print(
        f"    Validation : {len(val_dataset)} "
        f"({(1.0 - args.train_ratio) * 100:.1f}%)"
    )

    # ============================================================
    # OVERFIT DEBUG MODE
    #
    # Nếu --overfit được bật:
    # chỉ lấy một số ảnh từ TRAIN SET.
    #
    # Validation vẫn không được dùng để train.
    # ============================================================

    if args.overfit:

        n = min(
            args.overfit_images,
            len(train_dataset)
        )

        train_dataset = Subset(
            train_dataset,
            list(range(n))
        )

        print(
            "\n⚠️ OVERFIT DEBUG MODE"
        )

        print(
            f"Using only {n} labeled training images."
        )

        print(
            "Augmentation: OFF"
        )

        print(
            "Teacher: OFF"
        )

        print(
            "EMA: OFF"
        )

        print(
            "Unsupervised loss: OFF"
        )

        print(
            "SAM loss: OFF"
        )

    # ============================================================
    # TRAIN DATALOADER
    #
    # IMPORTANT:
    # Chỉ train trên train_dataset = 80%.
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=custom_collate
    )

    # ============================================================
    # VALIDATION DATALOADER
    #
    # Hiện tại train_supervised() chưa sử dụng validation loader.
    # Tuy nhiên ta tạo sẵn để pipeline có đúng 80/20 split.
    # Sau này có thể thêm validation loss / Dice / IoU.
    # ============================================================

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=custom_collate
    )

    # ============================================================
    # CHECK DATASET FORMAT
    # ============================================================

    print(
        "\n🔍 Checking training dataset format..."
    )

    first_batch = next(
        iter(train_loader)
    )

    print(
        f"Batch size: {len(first_batch)}"
    )

    print(
        f"Image shape: "
        f"{first_batch[0]['image'].shape}"
    )

    print(
        f"Instances: "
        f"{first_batch[0]['instances']}"
    )

    print(
        f"GT masks shape: "
        f"{first_batch[0]['instances'].gt_masks.shape}"
    )

    print(
        f"GT classes: "
        f"{first_batch[0]['instances'].gt_classes}"
    )

    # ============================================================
    # TEST NATIVE MASK2FORMER FORWARD
    # ============================================================

    print(
        "\n🔍 Testing native Mask2Former forward..."
    )

    model.train()

    with torch.enable_grad():

        test_batch = []

        for sample in first_batch:

            test_sample = {
                "image": sample["image"].to(args.device),
                "instances": sample["instances"].to(args.device),
                "height": sample["height"],
                "width": sample["width"],
                "image_id": sample["image_id"],
            }

            test_batch.append(
                test_sample
            )

        test_loss_dict = model(
            test_batch
        )

    print(
        "\nNative Mask2Former losses:"
    )

    for name, value in test_loss_dict.items():

        print(
            f"    {name}: "
            f"{value.detach().item():.6f}"
        )

    # ============================================================
    # OPTIMIZER PARAMETER GROUPS
    # ============================================================

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if name.startswith("backbone."):

            backbone_params.append(
                param
            )

        else:

            head_params.append(
                param
            )

    print(
        "\n🔧 Optimizer parameter groups:"
    )

    print(
        f"Backbone parameters: "
        f"{len(backbone_params)}"
    )

    print(
        f"Head parameters: "
        f"{len(head_params)}"
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": args.backbone_lr
            },
            {
                "params": head_params,
                "lr": args.lr
            }
        ],
        weight_decay=0.05
    )

    # ============================================================
    # SUPERVISED TRAINING
    #
    # IMPORTANT:
    #
    # train_supervised() receives ONLY train_loader.
    #
    # Therefore:
    #
    # L_sup is calculated on the 80% training set.
    #
    # Validation data is NOT used for backpropagation.
    # ============================================================

    print(
        "\n🚀 Starting supervised training..."
    )

    train_supervised(
        student_model=model,
        labeled_loader=train_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=args.device,
        max_grad_norm=1.0
    )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    torch.save(
        model.state_dict(),
        args.save_path
    )

    print(
        f"\n💾 Model saved to:\n"
        f"{args.save_path}"
    )

    print(
        "\n✅ SUPERVISED TRAINING COMPLETE"
    )

    print(
        "\n📌 Final dataset usage:"
    )

    print(
        f"    Training images    : {len(train_dataset)}"
    )

    print(
        f"    Validation images  : {len(val_dataset)}"
    )


if __name__ == "__main__":
    main()
