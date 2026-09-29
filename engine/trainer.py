import torch
import numpy as np
from models.losses import SupervisedLoss, UnsupervisedConsistencyLoss
from data.dataset import get_perturbations

def get_gaussian_rampup_weight(current_epoch, Tramp, lambda_max, gamma=5.0):
    if current_epoch >= Tramp:
        return lambda_max
    return lambda_max * np.exp(-gamma * (1 - current_epoch / Tramp) ** 2)

def update_ema_variables(student_model, teacher_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train_offline_semi_supervised(
    student_model, teacher_model, 
    labeled_loader, unlabeled_loader, 
    optimizer, epochs, device
):
    student_model.train()
    teacher_model.eval()
    
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    criterion_sup = SupervisedLoss().to(device)
    criterion_unsup = UnsupervisedConsistencyLoss().to(device)
    eta_s, eta_t = get_perturbations()
    
    T_ramp = 20
    lambda_max = 1.0
    alpha_ema = 0.99
    global_step = 0
    
    for epoch in range(epochs):
        lambda_t = get_gaussian_rampup_weight(epoch, T_ramp, lambda_max)
        
        for (x_l, y_l), (x_u, _) in zip(labeled_loader, unlabeled_loader):
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)
            optimizer.zero_grad()
            
            # [HACK]: Tạm thời tắt cờ training ở cấp cao nhất để Mask2Former 
            # chạy nhánh suy luận (trả về sem_seg thay vì tự tính loss)
            # nhưng các lớp bên dưới (BatchNorm, Dropout) vẫn giữ trạng thái train.
            student_model.training = False
            
            # [LUỒNG 1]: CÓ GIÁM SÁT
            outputs_l = student_model([{"image": img} for img in x_l])
            logits_l_mask = torch.stack([out["sem_seg"] for out in outputs_l])
            loss_sup = criterion_sup(logits_l_mask, y_l)
            
            # [LUỒNG 2]: PHI GIÁM SÁT
            x_u_s = eta_s(x_u)
            x_u_t = eta_t(x_u)
            
            outputs_u_student = student_model([{"image": img} for img in x_u_s])
            logits_u_student = torch.stack([out["sem_seg"] for out in outputs_u_student])
            
            # Bật lại cờ training
            student_model.training = True
            
            with torch.no_grad():
                outputs_u_teacher = teacher_model([{"image": img} for img in x_u_t])
                logits_u_teacher = torch.stack([out["sem_seg"] for out in outputs_u_teacher])
                
            loss_unsup = criterion_unsup(logits_u_student, logits_u_teacher)
            
            # [TỔI ƯU]
            total_loss = loss_sup + lambda_t * loss_unsup
            total_loss.backward()
            optimizer.step()
            
            update_ema_variables(student_model, teacher_model, alpha_ema, global_step)
            global_step += 1
            
        print(f"Epoch [{epoch}/{epochs}] | L_sup: {loss_sup.item():.4f} | L_unsup: {loss_unsup.item():.4f} | Lambda: {lambda_t:.4f}")
