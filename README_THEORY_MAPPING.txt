=============================================================================
TÀI LIỆU ÁNH XẠ MÃ NGUỒN VÀ LÝ THUYẾT (ARCHITECTURE THEORY MAPPING)
=============================================================================

Tài liệu này giải thích chi tiết vai trò của từng file `.py` trong thư mục 
`Thesis_source`, cũng như các hàm bên trong chúng tương ứng với phần nào, 
công thức nào trong file lý thuyết `Luận_văn__Phát_biểu_bài_toán.pdf`.

-----------------------------------------------------------------------------
1. data/dataset.py (Tương ứng: Phần 1 - Định nghĩa Không gian và Dữ liệu)
-----------------------------------------------------------------------------
- Vai trò: Xử lý toàn bộ các khái niệm về Không gian đầu vào (Input Space X), 
  Không gian đầu ra (Output Space Y) và Tập dữ liệu (Dataset).
- Hàm / Class chính:
  + `FracAtlasDataset`: Xây dựng tập dữ liệu. Trả về cấu trúc tensor X (ảnh gốc)
    và Y (mặt nạ nhị phân).
  + `get_perturbations()`: Tương ứng với Giai đoạn sinh bản đồ dự đoán độc lập 
    (Phương trình 12, 13). Tạo ra 2 hàm nhiễu loạn ngẫu nhiên độc lập (\eta_s 
    cho mạng Học viên và \eta_t cho mạng Giáo viên).
  + `custom_collate()`: Gom nhóm (batching) các mẫu dữ liệu, giả lập cho việc 
    tách luồng có nhãn (D_L) và không nhãn (D_U).

-----------------------------------------------------------------------------
2. models/specialist.py (Tương ứng: Phần 2 - Kiến trúc Mạng Chuyên gia S_\theta)
-----------------------------------------------------------------------------
- Vai trò: Định nghĩa "Hình hài" vật lý của mô hình, ánh xạ đầu vào X thành 
  ma trận xác suất [0, 1].
- Hàm / Class chính:
  + `EfficientViTDetectron2Wrapper` (Mục 2.1): Đại diện cho Bộ mã hóa ảnh 
    (Image Encoder). Trích xuất đặc trưng đa tỷ lệ F_enc = {1/16, 1/32, 1/64}. 
    Kế thừa kiến trúc Sandwich Layout và MHSA.
  + `setup_config()` (Mục 2.2 & 2.3): Cấu hình cho Pixel Decoder (sử dụng 
    MSDeformAttn) và Mask Decoder. Tạo ra bản đồ nhúng điểm ảnh E_pixel và 
    trích xuất ma trận Khóa/Giá trị (K, V).
  + `build_specialist_model()`: Lắp ráp Encoder và Decoder thành mạng S_\theta 
    hoàn chỉnh (Phương trình 7 - Final Mask Prediction).

-----------------------------------------------------------------------------
3. models/losses.py (Tương ứng: Phần 3 - Các Thành Phần Mất Mát)
-----------------------------------------------------------------------------
- Vai trò: Đóng gói các công thức tối ưu hóa (Loss function) thành các object 
  khả vi (differentiable) cho PyTorch.
- Hàm / Class chính:
  + `SupervisedLoss` và `DiceLoss` (Phương trình 10): Thành phần Mất mát có 
    giám sát (L_sup). Tối ưu hóa sai số nhị phân chéo (BCE) và mức độ chồng 
    lấp (Dice) trên tập dữ liệu có nhãn D_L.
  + `UnsupervisedConsistencyLoss` (Phương trình 14): Mất mát nhất quán (L_unsup). 
    Ép bản đồ xác suất của Học viên bám sát Giáo viên dưới 2 điều kiện nhiễu 
    khác nhau. Tính toán MSE (Sai số Toàn phương Trung bình) trên tập phi giám 
    sát D_U.

-----------------------------------------------------------------------------
4. engine/trainer.py (Tương ứng: Phần 3 - Giai đoạn Offline Bán giám sát)
-----------------------------------------------------------------------------
- Vai trò: Khung thuật toán Mean Teacher (SemiSAM+). Chịu trách nhiệm thực thi 
  Phương trình mục tiêu tổng quát (8). (Hiện tại đã tạm ẩn nhánh SAM).
- Hàm / Class chính:
  + `get_gaussian_rampup_weight()` (Phương trình 9): Khởi động mềm Gaussian. 
    Tính toán trọng số \lambda(t) theo thời gian, chống khuếch đại sai số ở 
    các epoch đầu.
  + `update_ema_variables()` (Phương trình 11): Cập nhật mạng Giáo viên. Áp 
    dụng phép Trung bình Trượt Mũ (EMA) từ trọng số của mạng Học viên để tạo 
    ra bản sao ổn định.
  + `train_offline_semi_supervised()`: Vòng lặp tối ưu Adam, chạy song song 2 
    luồng giám sát và phi giám sát, tính tổng loss và đạo hàm ngược \nabla_\theta.

-----------------------------------------------------------------------------
5. engine/inference.py (Tương ứng: Phần 4 - Giai đoạn Online)
-----------------------------------------------------------------------------
- Vai trò: Áp dụng luồng truyền thẳng (feed-forward) duy nhất cho ảnh mới để 
  suy luận thực tế.
- Hàm / Class chính:
  + `run_inference()` (Phương trình 20 & 21): Ánh xạ ảnh x_test thành bản đồ 
    xác suất không gian p_test. Dựa vào hàm kích hoạt Sigmoid và ngưỡng quyết 
    định để vẽ các mảng màu bán trong suốt (alpha overlay) chắp lên ảnh gốc.

-----------------------------------------------------------------------------
6. main.py (Entry Point)
-----------------------------------------------------------------------------
- Vai trò: Giao diện dòng lệnh (CLI). Định tuyến việc khởi chạy dự án, cho phép 
  chuyển đổi linh hoạt giữa việc Huấn luyện (mode: train) và Dự đoán thực tế 
  (mode: infer).
=============================================================================
