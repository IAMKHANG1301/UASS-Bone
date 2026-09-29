import os
import torch
import random
import matplotlib.pyplot as plt

def run_inference(model, dataset, device, output_filename="test_output_9_pairs.png"):
    model.eval()
    if len(dataset) == 0:
        print("❌ Không có ảnh nào để dự đoán.")
        return

    num_samples = min(9, len(dataset))
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"🚀 Đang chạy dự đoán trên {num_samples} ảnh ngẫu nhiên...")
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    fig.suptitle("Kết quả dự đoán: Ảnh gốc vs. Mặt nạ", fontsize=20, fontweight='bold')
    
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            row = i // 3
            col_base = (i % 3) * 2
            
            sample = dataset[idx]
            sample_input = {
                "image": sample["image"].to(device),
                "height": sample["height"],
                "width": sample["width"]
            }
            
            outputs = model([sample_input])
            orig_img = sample["image"].cpu().numpy().transpose(1, 2, 0) / 255.0
            mask_tensor = outputs[0]["sem_seg"][0].cpu().numpy()
            
            ax_img = axes[row, col_base]
            ax_img.imshow(orig_img)
            ax_img.axis('off')
            ax_img.set_title(f"Ảnh gốc (ID: {idx})", fontsize=12)
            
            ax_mask = axes[row, col_base + 1]
            ax_mask.imshow(mask_tensor, cmap='jet', alpha=0.8)
            ax_mask.axis('off')
            ax_mask.set_title(f"Dự đoán Mask", fontsize=12)
            
    for j in range(num_samples * 2, 18):
        r, c = j // 6, j % 6
        axes[r, c].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=150)
    print(f"✅ DỰ ĐOÁN THÀNH CÔNG! Đã lưu kết quả tại '{output_filename}'")
