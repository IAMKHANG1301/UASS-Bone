import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FracAtlasDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Tạo 2 mảng để lưu đường dẫn đầy đủ và tên file
        self.image_paths = []
        self.image_names = []
        
        # Quét đệ quy chui vào cả thư mục Fractured và Non_fractured
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, f))
                    self.image_names.append(f) # Lưu tên file để tìm Mask tương ứng
                    
    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        mask = None
        if self.mask_dir is not None:
            # Mask_dir của bạn là thư mục phẳng, nên chỉ cần ghép với img_name
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                try:
                    mask = Image.open(mask_path).convert('L')
                    mask = np.array(mask)
                    mask = (mask > 127).astype(np.int64) 
                    mask = torch.as_tensor(mask)
                except Exception:
                    mask = torch.zeros((image.size[1], image.size[0]), dtype=torch.int64)
                
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image)
            
        dataset_dict = {
            "image": torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32),
            "height": image.shape[0],
            "width": image.shape[1],
            "image_id": idx,
        }
        
        if mask is not None:
            dataset_dict["sem_seg"] = mask
        return dataset_dict

def get_perturbations():
    """Tạo 2 hàm nhiễu độc lập (eta_s, eta_t) cho Học viên và Giáo viên"""
    eta_s = T.Compose([T.RandomApply([T.ColorJitter(0.2, 0.2)], p=0.5), T.RandomAffine(degrees=10, translate=(0.05, 0.05))])
    eta_t = T.Compose([T.RandomApply([T.ColorJitter(0.1, 0.1)], p=0.5), T.RandomAffine(degrees=5, translate=(0.02, 0.02))])
    return eta_s, eta_t

def custom_collate(batch):
    images = torch.stack([item["image"] for item in batch])
    if "sem_seg" in batch[0]:
        masks = torch.stack([item["sem_seg"] for item in batch]).unsqueeze(1)
    else:
        masks = torch.zeros(len(batch), 1, images.shape[2], images.shape[3])
    return images, masks
