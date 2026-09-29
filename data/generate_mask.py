import os
import json
import numpy as np
import cv2
from PIL import Image, ImageFile

# Cờ quan trọng nhất: Ép hệ thống phớt lờ lỗi cụt đuôi file JPEG của FracAtlas
ImageFile.LOAD_TRUNCATED_IMAGES = True

def generate_masks(json_path, non_fractured_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Đang lưu mặt nạ tại: {output_dir}")

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. Tạo mặt nạ cho ảnh CÓ vết nứt (Fractured)
    # Giai đoạn này cực kỳ an toàn vì ta lấy width/height từ file JSON, hoàn toàn KHÔNG chạm vào file ảnh gốc.
    annotations_map = {}
    if "annotations" in coco_data:
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_map:
                annotations_map[img_id] = []
            if "segmentation" in ann and not ann.get("iscrowd", 0):
                annotations_map[img_id].extend(ann["segmentation"])

    generated_files = set()
    print("🚀 Đang vẽ mặt nạ cho thư mục Fractured...")
    for img_info in coco_data["images"]:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        if img_id in annotations_map:
            for seg in annotations_map[img_id]:
                poly = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], 255)
                
        cv2.imwrite(os.path.join(output_dir, file_name), mask)
        generated_files.add(file_name)

    # 2. Tạo mặt nạ cho ảnh KHÔNG có vết nứt (Non_fractured)
    print("🚀 Đang tạo mặt nạ rỗng cho thư mục Non_fractured...")
    if os.path.exists(non_fractured_dir):
        for file in os.listdir(non_fractured_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file not in generated_files:
                img_path = os.path.join(non_fractured_dir, file)
                try:
                    # ĐIỂM CHỐNG LỖI: Image.open() ở đây chỉ đọc Header để lấy kích thước,
                    # không hề decode mảng pixel nên sẽ lướt qua lỗi "Premature end of JPEG" một cách mượt mà.
                    with Image.open(img_path) as img:
                        w, h = img.size
                        
                    blank_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.imwrite(os.path.join(output_dir, file), blank_mask)
                    generated_files.add(file)
                except Exception as e:
                    print(f"⚠️ Không thể đọc file {file}: {e}")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {non_fractured_dir}")
                    
    print(f"✅ HOÀN TẤT! Đã tạo thành công {len(generated_files)} ảnh mặt nạ.")

if __name__ == "__main__":
    # ĐƯỜNG DẪN DÀNH CHO KAGGLE
    JSON_PATH = "/kaggle/input/datasets/mahmudulhasantasin/fracatlas-original-dataset/FracAtlas/Annotations/COCO JSON/COCO_fracture_masks.json" 
    NON_FRACTURED_DIR = "/kaggle/input/datasets/mahmudulhasantasin/fracatlas-original-dataset/FracAtlas/images/Non_fractured"
    OUTPUT_MASKS_DIR = "/kaggle/working/masks"
    

    # Đường dẫn dành cho môi trường laptop cá nhân
    # JSON_PATH = "../FracAtlas/Annotations/COCO JSON/COCO_fracture_masks.json" 
    # NON_FRACTURED_DIR = "../FracAtlas/images/Non_fractured"
    # OUTPUT_MASKS_DIR = "./Masks"
    
    generate_masks(JSON_PATH, NON_FRACTURED_DIR, OUTPUT_MASKS_DIR)