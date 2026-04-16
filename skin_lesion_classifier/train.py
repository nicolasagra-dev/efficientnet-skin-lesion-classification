from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import TrainConfig
from .data import create_dataloaders, prepare_data
from .model import (
    build_model,
    create_criterion,
    freeze_feature_extractor,
    unfreeze_last_blocks,
)
from .utils import (
    configure_device,
    generate_grad_cam,
    maybe_show_or_save,
    plot_confusion_matrix,
    plot_training_history,
    set_seed,
    visualize_frequency,
    visualize_samples,
)


def build_tta_variants(images, passes: int):
    variants = [
        images,
        torch.flip(images, dims=[3]),
        torch.flip(images, dims=[2]),
        torch.flip(images, dims=[2, 3]),
    ]
    return variants[: max(1, min(passes, len(variants)))]


def predict_probabilities(model, images, tta_passes=1):
    variants = build_tta_variants(images, tta_passes)
    probs_sum = None

    for variant in variants:
        logits = model(variant)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs_sum = probs if probs_sum is None else probs_sum + probs

    return probs_sum / len(variants)


def run_epoch(model, loader, criterion, optimizer, device, config: TrainConfig, scaler=None, training=False, tta_passes=1):
    if training:
        if optimizer is None:
            raise ValueError("Optimizer is required during training.")
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    valid_batches = 0
    all_probs = []
    all_preds = []
    all_targets = []

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=device.type == "cuda" and config.use_amp):
                logits = model(images)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                loss = criterion(logits, targets)

            if training:
                if not torch.isfinite(loss):
                    print("AVISO: batch ignorado por perda nao finita durante o treino.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss_to_backprop = loss / config.gradient_accumulation_steps
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

                should_step = (
                    batch_index % config.gradient_accumulation_steps == 0
                    or batch_index == len(loader)
                )
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                probs = torch.softmax(logits, dim=1)
            else:
                probs = (
                    predict_probabilities(model, images, tta_passes=tta_passes)
                    if tta_passes > 1
                    else torch.softmax(logits, dim=1)
                )

        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == targets).sum().item()
        total += images.size(0)
        valid_batches += 1

        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    if valid_batches == 0 or total == 0:
        raise RuntimeError("Nenhum batch valido foi processado nesta etapa.")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return epoch_loss, epoch_acc, all_probs, all_preds, all_targets


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, class_names, stage_name, config: TrainConfig) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "image_size": (config.img_height, config.img_width),
            "stage_name": stage_name,
            "model_name": config.model_name,
        },
        path,
    )


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    stage_name,
    epochs,
    learning_rate,
    best_model_path,
    best_val_acc,
    history,
    config: TrainConfig,
):
    optimizer = Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda" and config.use_amp)
    epochs_without_improvement = 0

    print(f"\n[7] STARTING {stage_name.upper()}...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config,
            scaler=scaler,
            training=True,
            tta_passes=1,
        )
        val_loss, val_acc, _, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            config,
            scaler=scaler,
            training=False,
            tta_passes=1,
        )

        history["stage"].append(stage_name)
        history["epoch"].append(len(history["epoch"]) + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        print(
            f"{stage_name} | epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc + config.early_stopping_min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                history["epoch"][-1],
                best_val_acc,
                config.class_names,
                stage_name,
                config,
            )
            print(f"Best model saved to: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No meaningful improvement for {epochs_without_improvement} epoch(s). "
                f"Patience: {config.early_stopping_patience}"
            )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping acionado em {stage_name}.")
            break

    return best_val_acc, history


def evaluate_model(model, test_loader, criterion, device, class_to_idx, config: TrainConfig) -> None:
    print("\n[8] EVALUATION...")
    checkpoint = load_checkpoint(config.best_model_path, model, device)
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    print(f"Best stage: {checkpoint.get('stage_name', 'unknown')}")

    test_loss, test_acc, test_probs, test_preds, test_targets = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        config=config,
        scaler=None,
        training=False,
        tta_passes=config.tta_passes,
    )
    test_probs = np.nan_to_num(test_probs, nan=0.0, posinf=1.0, neginf=0.0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    target_names = [idx_to_class[index] for index in range(len(idx_to_class))]

    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names, zero_division=0))

    cm_matrix = confusion_matrix(test_targets, test_preds)
    plot_confusion_matrix(cm_matrix, target_names, config, normalize=True, title="Confusion Matrix")

    print("\nGenerating ROC curve data...")
    lb = LabelBinarizer()
    lb.fit(range(len(target_names)))
    y_test_bin = lb.transform(test_targets)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    for class_index, class_name in enumerate(target_names):
        if np.unique(y_test_bin[:, class_index]).size < 2:
            print(f"ROC ignorada para {class_name} por falta de exemplos suficientes.")
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_index], test_probs[:, class_index])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) - Multiclass")
    plt.legend(loc="lower right")
    maybe_show_or_save(
        config.plots_dir / "roc_curve.png",
        show_plots=config.show_plots,
        save_plots=config.save_plots,
    )

    if config.run_grad_cam:
        print("\n[9] GENERATING GRAD-CAM EXAMPLE...")
        sample_images, sample_targets = next(iter(test_loader))
        generate_grad_cam(
            model,
            sample_images[0],
            device,
            target_names,
            config,
            true_label=sample_targets[0].item(),
        )
        print("Grad-CAM salvo em plots/grad_cam_example.png")
    else:
        print("\n[9] Grad-CAM desativado.")

    print("O treino e a inferencia usam a GTX 1650 via CUDA quando disponivel.")


def run_training_pipeline(config: TrainConfig | None = None) -> None:
    config = config or TrainConfig()

    set_seed(config.seed)
    device = configure_device(config)
    df, class_to_idx = prepare_data(config)

    if config.save_plots or config.show_plots:
        visualize_frequency(df, config)
        visualize_samples(df, config)

    print("\n[4] SPLITTING DATA (TRAIN/VAL/TEST)...")
    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["cell_type"], random_state=config.seed)
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["cell_type"],
        random_state=config.seed,
    )

    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, device, config)
    model = build_model(device, config)
    criterion = create_criterion(train_df, device, config)

    history = {"stage": [], "epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    freeze_feature_extractor(model)
    best_val_acc, history = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        stage_name="head_training",
        epochs=config.head_epochs,
        learning_rate=config.head_learning_rate,
        best_model_path=config.best_model_path,
        best_val_acc=best_val_acc,
        history=history,
        config=config,
    )

    load_checkpoint(config.best_model_path, model, device)
    unfreeze_last_blocks(model, config.unfreeze_last_blocks)
    best_val_acc, history = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        stage_name="fine_tuning",
        epochs=config.finetune_epochs,
        learning_rate=config.finetune_learning_rate,
        best_model_path=config.best_model_path,
        best_val_acc=best_val_acc,
        history=history,
        config=config,
    )

    if config.save_plots or config.show_plots:
        plot_training_history(history, config)

    evaluate_model(model, test_loader, criterion, device, class_to_idx, config)
