import torch
import numpy as np
import torch.nn.functional as F
from models.losses import SupervisedLoss, UnsupervisedConsistencyLoss, SAMGuidedSoftLoss
from data.dataset import get_perturbations

def get_sam_pseudo_labels_and_uncertainty(sam_model, images, student_masks, n_prompts=4):
    """
    Input:
    - images: [B, 3, H, W], torch.float32, on GPU
    - student_masks: [B, 1, H, W], torch.float32, on GPU (đã qua sigmoid/clamp)
    Output:
    - pseudo_labels: [B, 1, H, W], torch.float32
    - uncertainty_maps: [B, 1, H, W], torch.float32
    """
    B, C, H, W = images.shape
    device = images.device
    
    batch_pseudo_labels = []
    batch_uncertainties = []
    
    with torch.no_grad(): # CHÚ Ý: Chặn toàn bộ đạo hàm
        prompt_size = (256, 256)
        sam_image_size = (1024, 1024)
        
        resized_masks = F.interpolate(student_masks, size=prompt_size, mode='bilinear', align_corners=False)
        resized_images = F.interpolate(images, size=sam_image_size, mode='bilinear', align_corners=False)
        
        for b in range(B):
            curr_image = resized_images[b].unsqueeze(0) # [1, 3, 1024, 1024]
            curr_mask = resized_masks[b].unsqueeze(0) # [1, 1, 256, 256]
            
            # 1. Encode từng ảnh một để tránh tràn VRAM trên GPU
            curr_embedding = sam_model.image_encoder(curr_image) # [1, 256, 64, 64]
            
            # 2. Mở rộng prompt thành n_prompts bản sao và thêm nhiễu
            curr_mask_expanded = curr_mask.repeat(n_prompts, 1, 1, 1) # [n_prompts, 1, 256, 256]
            noise = torch.randn_like(curr_mask_expanded) * 0.1
            prompt_masks = torch.clamp(curr_mask_expanded + noise, 0.0, 1.0)
            
            # 3. Chạy Prompt Encoder cho toàn bộ n_prompts cùng lúc
            sparse_embeds, dense_embeds = sam_model.prompt_encoder(
                points=None, boxes=None, masks=prompt_masks
            )
            
            # 4. Giải mã: SAM sẽ tự động nhân bản curr_embedding lên n_prompts lần
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=curr_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeds,
                dense_prompt_embeddings=dense_embeds,
                multimask_output=False,
            )
            
            # 5. Phục hồi kích thước gốc
            mask_up = F.interpolate(low_res_masks, size=(H, W), mode='bilinear', align_corners=False)
            sam_masks = torch.sigmoid(mask_up) # [n_prompts, 1, H, W]
            
            # 6. Gom kết quả của ảnh b
            batch_pseudo_labels.append(torch.mean(sam_masks, dim=0, keepdim=True)) # [1, 1, H, W]
            batch_uncertainties.append(torch.var(sam_masks, dim=0, keepdim=True)) # [1, 1, H, W]
            
    # Nối lại thành Batch hoàn chỉnh
    pseudo_labels = torch.cat(batch_pseudo_labels, dim=0) # [B, 1, H, W]
    uncertainty_maps = torch.cat(batch_uncertainties, dim=0) # [B, 1, H, W]
    
    return pseudo_labels, uncertainty_maps

def get_gaussian_rampup_weight(current_epoch, Tramp, lambda_max, gamma=5.0):
    if current_epoch >= Tramp:
        return lambda_max
    return lambda_max * np.exp(-gamma * (1 - current_epoch / Tramp) ** 2)

def update_ema_variables(student_model, teacher_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # Update parameters
    for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    # Update buffers (BN stats, v.v.)
    for ema_buffer, buffer in zip(teacher_model.buffers(), student_model.buffers()):
        ema_buffer.data.copy_(buffer.data)

def train_offline_semi_supervised(
    student_model, teacher_model, sam_model,
    labeled_loader, unlabeled_loader, 
    optimizer, epochs, device, loss_type="bce_dice"
):
    student_model.train()
    teacher_model.eval()
    
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    criterion_sup = SupervisedLoss(loss_type=loss_type).to(device)
    criterion_unsup = UnsupervisedConsistencyLoss().to(device)
    criterion_sam = SAMGuidedSoftLoss().to(device)
    eta_s, eta_t = get_perturbations()
    
    T_ramp = 20
    lambda_max = 1.0
    beta_max = 0.5
    alpha_ema = 0.99
    global_step = 0
    accumulation_steps = 4
    
    for epoch in range(epochs):
        # KHÓA CỨNG 2 NHÁNH PHI GIÁM SÁT VÀ SAM
        lambda_t = 0.0  
        beta_t = 0.0    
        
        optimizer.zero_grad()
        
        for i, ((x_l, y_l), (x_u, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)
            
            # [LUỒNG 1]: TÍNH TOÁN L_SUP
            student_model.training = False # Force output format to be predictions
            outputs_l = student_model([{"image": img} for img in x_l])
            student_model.training = True # Restore training mode
            logits_l_mask = torch.stack([out["sem_seg"] for out in outputs_l])
            loss_sup = criterion_sup(logits_l_mask, y_l)
            
            # Tắt hoàn toàn luồng tính loss_unsup và loss_sam
            loss_unsup = torch.tensor(0.0).to(device)
            loss_sam = torch.tensor(0.0).to(device)
            
            # [TỔNG HỢP L_SUP]
            total_loss = loss_sup / accumulation_steps
            total_loss.backward()
            
            # [IN DEBUG CHẨN ĐOÁN VẬT LÝ Ở BATCH ĐẦU TIÊN]
            if i == 0: 
                with torch.no_grad():
                    preds_prob = torch.sigmoid(logits_l_mask)
                    
                print(f"\nEpoch [{epoch}/{epochs}] --- CHẨN ĐOÁN VẬT LÝ MASK2FORMER ---")
                print(f"1. Thang đo ảnh đầu vào (Image Range): Min = {x_l.min().item():.4f} | Max = {x_l.max().item():.4f}")
                if x_l.max() <= 1.0:
                    print("   -> ⚠️ BÁO ĐỘNG: Mask2Former có thể đang bị mù do ảnh dải 0-1. Thử nhân x_l * 255.0 trước khi nạp vào model!")
                    
                print(f"2. Kích thước Tensor (Shapes):")
                print(f"   - Dự đoán (logits_l_mask): {logits_l_mask.shape}")
                print(f"   - Nhãn thật (y_l): {y_l.shape}")
                if logits_l_mask.shape[1] != y_l.shape[1]:
                    print("   -> ⚠️ BÁO ĐỘNG: Lỗi chênh lệch số kênh! Hàm Loss đang bị tính sai (Broadcast Bug).")
                    
                print(f"3. Lỗi hội tụ (Loss): {loss_sup.item():.4f}")
                print(f"4. Thống kê dự đoán (Preds): Max = {preds_prob.max().item():.4f} | Mean = {preds_prob.mean().item():.4f}")
                print("---------------------------------------------------------")
                
            # [CẬP NHẬT TRỌNG SỐ GRADIENT ACCUMULATION]
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(labeled_loader):
                optimizer.step()
                optimizer.zero_grad()
                update_ema_variables(student_model, teacher_model, alpha_ema, global_step)
                global_step += 1
                
        print(f"Epoch [{epoch}/{epochs}] | L_sup: {loss_sup.item():.4f} | L_unsup: {loss_unsup.item():.4f} | L_sam: {loss_sam.item():.4f} | Lambda: {lambda_t:.4f}")
