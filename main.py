import os
import sys
import argparse
import torch
import copy
from torch.utils.data import DataLoader

sys.path.append('/kaggle/working/efficientvit')
sys.path.append('/kaggle/working/Mask2Former')

from detectron2.data import DatasetCatalog, MetadataCatalog 

from data.dataset import FracAtlasDataset, custom_collate
from models.specialist import setup_config, build_specialist_model
from engine.trainer import train_offline_semi_supervised
from engine.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="Chạy Pipeline Specialist cho luận văn (Huấn luyện & Suy luận).")
    parser.add_argument("--mode", type=str, default="train", choices=["infer", "train"], help="Chế độ chạy")
    parser.add_argument("--image_dir", type=str, required=True, help="Đường dẫn đến thư mục chứa ảnh X-quang")
    parser.add_argument("--mask_dir", type=str, default=None, help="Đường dẫn đến thư mục chứa ảnh mask nhãn (tuỳ chọn)")
    parser.add_argument("--weight_path", type=str, default="specialist_teacher_model.pth", help="Đường dẫn lưu/nạp trọng số mô hình (.pth)")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epochs huấn luyện")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị tính toán")
    
    args = parser.parse_args()
    
    if "fracatlas_train" not in DatasetCatalog.list():
        DatasetCatalog.register("fracatlas_train", lambda: [])
        MetadataCatalog.get("fracatlas_train").set(
            thing_classes=["fracture"], 
            stuff_classes=["background"], 
            ignore_label=255
        )
        
    print(f"🔄 Khởi tạo Mạng Specialist (Student) trên thiết bị: {args.device.upper()}...")
    cfg = setup_config()
    specialist_model = build_specialist_model(cfg)
    specialist_model.to(args.device)
    
    print(f"📂 Đang nạp dataset từ: {args.image_dir}")
    full_dataset = FracAtlasDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)
    print(f"📊 Tổng số ảnh tìm thấy: {len(full_dataset)}")
    
    if args.mode == "train":
        print("\n=======================================================")
        print("🚀 BẮT ĐẦU GIAI ĐOẠN 3: HUẤN LUYỆN BÁN GIÁM SÁT (OFFLINE)")
        print("=======================================================")
        
        teacher_model = copy.deepcopy(specialist_model)
        teacher_model.to(args.device)
        
        # NOTE: Để demo, dùng chung full_dataset cho cả tập có nhãn (D_L) và không nhãn (D_U).
        # Trong thực tế, bạn sẽ chia split dataset ra làm 2 phần.
        train_loader = DataLoader(full_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)
        unlabeled_loader = DataLoader(full_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)
        
        optimizer = torch.optim.Adam(specialist_model.parameters(), lr=1e-4)
        
        train_offline_semi_supervised(
            student_model=specialist_model, 
            teacher_model=teacher_model, 
            labeled_loader=train_loader, 
            unlabeled_loader=unlabeled_loader, 
            optimizer=optimizer, 
            epochs=args.epochs, 
            device=args.device
        )
        
        # LƯU TRỮ MODEL
        # Theo lý thuyết Mean Teacher, mạng Giáo viên có tính ổn định cao hơn Học viên
        torch.save(teacher_model.state_dict(), args.weight_path)
        print(f"💾 Đã lưu trọng số mạng Giáo viên (Teacher) tại: {args.weight_path}")
        print("✅ Hoàn tất Giai đoạn Offline!")
        
    elif args.mode == "infer":
        print("\n=======================================================")
        print("🚀 BẮT ĐẦU GIAI ĐOẠN 4: CHẨN ĐOÁN (ONLINE INFERENCE)")
        print("=======================================================")
        
        if os.path.exists(args.weight_path):
            print(f"📥 Đang nạp trọng số đã huấn luyện từ: {args.weight_path}")
            specialist_model.load_state_dict(torch.load(args.weight_path, map_location=args.device, weights_only=True))
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy {args.weight_path}. Đang chạy dự đoán bằng trọng số khởi tạo ngẫu nhiên!")
            
        run_inference(specialist_model, full_dataset, args.device)

if __name__ == "__main__":
    main()
