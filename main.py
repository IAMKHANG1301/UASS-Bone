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
from torch.utils.data import DataLoader, Subset
from detectron2.data import DatasetCatalog, MetadataCatalog
from data.dataset import FracAtlasDataset, custom_collate
from models.specialist import setup_config, build_specialist_model
from engine.trainer import train_supervised

def main():
    parser = argparse.ArgumentParser(description="UASS-Bone supervised Mask2Former training")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone_lr", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overfit", action="store_true", help="Overfit a very small labeled subset to verify supervised convergence.")
    parser.add_argument("--overfit_images", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="specialist_supervised.pth")

    args = parser.parse_args()

    if "fracatlas_train" not in DatasetCatalog.list():
        DatasetCatalog.register("fracatlas_train", lambda: [])
        MetadataCatalog.get("fracatlas_train").set(
            thing_classes=["fracture"],
            stuff_classes=["background"],
            ignore_label=255
        )

    print("\n🔧 Building Mask2Former configuration...")
    cfg = setup_config()

    print("🧠 Building EfficientViT-B0 + Mask2Former...")
    model = build_specialist_model(cfg)
    model.to(args.device)

    print(f"\n📂 Image directory:\n{args.image_dir}")
    print(f"📂 Mask directory:\n{args.mask_dir}")

    dataset = FracAtlasDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        transform=None
    )

    print(f"\n📊 Total images: {len(dataset)}")

    if args.overfit:
        n = min(args.overfit_images, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        print("\n⚠️ OVERFIT DEBUG MODE")
        print(f"Using only {n} labeled images.")
        print("Augmentation: OFF\nTeacher: OFF\nEMA: OFF\nUnsupervised loss: OFF\nSAM loss: OFF")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=custom_collate
    )

    print("\n🔍 Checking dataset format...")
    first_batch = next(iter(train_loader))
    print(f"Batch size: {len(first_batch)}")
    print(f"Image shape: {first_batch[0]['image'].shape}")
    print(f"Instances: {first_batch[0]['instances']}")
    print(f"GT masks shape: {first_batch[0]['instances'].gt_masks.shape}")
    print(f"GT classes: {first_batch[0]['instances'].gt_classes}")

    print("\n🔍 Testing native Mask2Former forward...")
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
            test_batch.append(test_sample)

        test_loss_dict = model(test_batch)

    print("\nNative Mask2Former losses:")
    for name, value in test_loss_dict.items():
        print(f"    {name}: {value.detach().item():.6f}")

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    print("\n🔧 Optimizer parameter groups:")
    print(f"Backbone parameters: {len(backbone_params)}")
    print(f"Head parameters: {len(head_params)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr}
        ],
        weight_decay=0.05
    )

    print("\n🚀 Starting supervised training...")
    train_supervised(
        student_model=model,
        labeled_loader=train_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=args.device,
        max_grad_norm=1.0
    )

    torch.save(model.state_dict(), args.save_path)
    print(f"\n💾 Model saved to:\n{args.save_path}")
    print("\n✅ SUPERVISED TRAINING COMPLETE")

if __name__ == "__main__":
    main()
