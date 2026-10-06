import torch


def train_supervised(
    student_model,
    labeled_loader,
    optimizer,
    epochs,
    device,
    max_grad_norm=1.0
):

    print()
    print("=" * 70)
    print("SUPERVISED-ONLY MASK2FORMER TRAINING")
    print("=" * 70)
    print()

    student_model.to(device)

    total_iterations = (
        epochs
        * len(labeled_loader)
    )

    current_iteration = 0

    poly_power = 0.9

    for epoch in range(epochs):

        student_model.train()

        epoch_loss = 0.0
        num_batches = 0

        last_loss_dict = None
        last_grad_norm = 0.0

        for batch in labeled_loader:

            # --------------------------------------------
            # MOVE TO DEVICE
            # --------------------------------------------

            for sample in batch:

                sample["image"] = (
                    sample["image"].to(device)
                )

                sample["instances"] = (
                    sample["instances"].to(device)
                )

            # --------------------------------------------
            # ZERO GRAD
            # --------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # --------------------------------------------
            # POLY LR
            # --------------------------------------------

            progress = (
                current_iteration
                /
                max(total_iterations - 1, 1)
            )

            lr_multiplier = (
                (1.0 - progress)
                ** poly_power
            )

            for param_group in optimizer.param_groups:
                
                # Check if 'initial_lr' is not set, set it
                if "initial_lr" not in param_group:
                    param_group["initial_lr"] = param_group["lr"]

                param_group["lr"] = (
                    param_group["initial_lr"]
                    * lr_multiplier
                )

            # --------------------------------------------
            # NATIVE MASK2FORMER
            # --------------------------------------------

            loss_dict = student_model(
                batch
            )

            last_loss_dict = loss_dict

            # --------------------------------------------
            # TOTAL LOSS
            # --------------------------------------------

            loss_sup = sum(
                loss
                for loss in loss_dict.values()
                if torch.is_tensor(loss)
            )

            # --------------------------------------------
            # CHECK
            # --------------------------------------------

            if not torch.isfinite(loss_sup):

                print(
                    "❌ Non-finite L_sup:"
                )

                print(loss_dict)

                raise RuntimeError(
                    "L_sup became NaN/Inf"
                )

            # --------------------------------------------
            # BACKPROP
            # --------------------------------------------

            loss_sup.backward()

            # --------------------------------------------
            # GRADIENT CLIPPING
            # --------------------------------------------

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(),
                    max_grad_norm
                )
            )

            last_grad_norm = float(
                grad_norm
            )

            # --------------------------------------------
            # OPTIMIZER
            # --------------------------------------------

            optimizer.step()

            current_iteration += 1

            epoch_loss += (
                loss_sup.detach().item()
            )

            num_batches += 1

        # ----------------------------------------------
        # EPOCH RESULT
        # ----------------------------------------------

        avg_loss = (
            epoch_loss
            /
            max(num_batches, 1)
        )

        current_lrs = [
            param_group["lr"]
            for param_group
            in optimizer.param_groups
        ]

        print(
            f"Epoch "
            f"[{epoch + 1:03d}/{epochs:03d}] "
            f"| L_sup: {avg_loss:.6f}"
        )

        if len(current_lrs) >= 2:
            print(
                f"    "
                f"backbone_lr={current_lrs[0]:.8e} | "
                f"head_lr={current_lrs[1]:.8e}"
            )
        else:
            print(
                f"    "
                f"lr={current_lrs[0]:.8e}"
            )

        if last_loss_dict is not None:

            loss_summary = []

            for name, value in (
                last_loss_dict.items()
            ):

                if torch.is_tensor(value):

                    loss_summary.append(
                        f"{name}="
                        f"{value.detach().item():.4f}"
                    )

            print(
                "    "
                + " | ".join(loss_summary)
            )

            print(
                f"    pre-clip grad_norm="
                f"{last_grad_norm:.4f}"
            )
            
        print()


    print("=" * 70)
    print("✅ Supervised training finished.")
    print("=" * 70)


def train_offline_semi_supervised(
    *args,
    **kwargs
):

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
