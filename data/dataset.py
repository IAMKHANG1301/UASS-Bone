import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from detectron2.structures import BitMasks, Instances


def mask_to_instances(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    binary_mask = mask > 0
    height, width = binary_mask.shape
    instances = Instances(image_size=(height, width))

    if binary_mask.sum() == 0:
        instances.gt_classes = torch.empty((0,), dtype=torch.int64)
        instances.gt_masks = BitMasks(torch.empty((0, height, width), dtype=torch.bool))
        return instances

    gt_mask = torch.from_numpy(binary_mask.astype(np.bool_)).unsqueeze(0)
    instances.gt_classes = torch.zeros((1,), dtype=torch.int64)
    instances.gt_masks = BitMasks(gt_mask)

    return instances

class FracAtlasDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = []
        self.image_names = []

        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(root, f))
                    self.image_names.append(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]

        image = Image.open(img_path).convert("RGB")
        target_size = (512, 512)
        image = image.resize(target_size, resample=Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.uint8)

        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert("L")
                    mask_img = mask_img.resize(target_size, resample=Image.NEAREST)
                    mask_np = np.asarray(mask_img)
                    mask = (mask_np > 127).astype(np.uint8)
                except Exception as e:
                    print(f"⚠️ Cannot read mask {mask_path}: {e}")
                    mask = np.zeros((512, 512), dtype=np.uint8)
            else:
                raise FileNotFoundError(f"Mask not found for image:\nImage: {img_path}\nExpected mask: {mask_path}")

        if self.transform is not None:
            image_np = self.transform(image_np)

        if isinstance(image_np, np.ndarray):
            image_tensor = torch.from_numpy(np.ascontiguousarray(image_np.transpose(2, 0, 1))).float()
        else:
            image_tensor = image_np.float()

        dataset_dict = {
            "image": image_tensor,
            "height": 512,
            "width": 512,
            "image_id": idx,
        }

        if mask is not None:
            dataset_dict["instances"] = mask_to_instances(mask)

        return dataset_dict


def custom_collate(batch):
    return batch


def get_perturbations():
    eta_s = T.Compose([T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(p=0.2)])
    eta_t = T.Compose([T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8)])
    return eta_s, eta_t
