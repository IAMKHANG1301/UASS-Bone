import torch

def train_supervised(student_model, labeled_loader, optimizer, epochs, device, max_grad_norm=1.0):
    print()
    print("=" * 70)
    print("SUPERVISED-ONLY MASK2FORMER TRAINING")
    print("=" * 70)
    print()

    student_model.to(device)

    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0.0
        num_batches = 0
        last_loss_dict = None

        for batch_idx, batch in enumerate(labeled_loader):
            for sample in batch:
                sample["image"] = sample["image"].to(device)
                sample["instances"] = sample["instances"].to(device)

            optimizer.zero_grad(set_to_none=True)

            loss_dict = student_model(batch)
            last_loss_dict = loss_dict

            loss_sup = sum(loss for loss in loss_dict.values() if torch.is_tensor(loss))

            if not torch.isfinite(loss_sup):
                print("❌ Non-finite supervised loss!")
                print(loss_dict)
                raise RuntimeError("L_sup became NaN/Inf.")

            loss_sup.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += loss_sup.detach().item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch [{epoch + 1:03d}/{epochs:03d}] | L_sup: {avg_loss:.6f}")

        if last_loss_dict is not None:
            loss_summary = []
            for name, value in last_loss_dict.items():
                if torch.is_tensor(value):
                    loss_summary.append(f"{name}={value.detach().item():.4f}")
            print("    " + " | ".join(loss_summary))
            print(f"    grad_norm={float(grad_norm):.4f}")
        print()

    print("=" * 70)
    print("✅ Supervised training finished.")
    print("=" * 70)


def train_offline_semi_supervised(*args, **kwargs):
    raise RuntimeError(
        "\n"
        "train_offline_semi_supervised() is intentionally "
        "disabled during the supervised convergence test.\n\n"
        "Use train_supervised() first.\n\n"
        "Reason:\n"
        "You must prove that EfficientViT + Mask2Former "
        "can converge with L_sup before adding "
        "Teacher/EMA/L_unsup/SAM.\n"
    )
