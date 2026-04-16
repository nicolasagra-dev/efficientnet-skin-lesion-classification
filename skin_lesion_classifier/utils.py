from __future__ import annotations

import itertools
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(config: TrainConfig) -> None:
    config.image_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir.mkdir(parents=True, exist_ok=True)
    if config.save_plots:
        config.plots_dir.mkdir(parents=True, exist_ok=True)


def configure_device(config: TrainConfig) -> torch.device:
    print("=" * 60)
    print("CONFIGURANDO DISPOSITIVO...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detectada: {gpu_name}")
        print(f"Memoria total: {total_memory_gb:.2f} GB")
        print("PyTorch vai treinar usando CUDA.")
        print(f"Modelo: {config.model_name}")
        print(f"Resolucao: {config.img_height}x{config.img_width}")
        print(f"Batch real: {config.batch_size}")
        print(f"Acumulacao de gradiente: {config.gradient_accumulation_steps}")
        print("AMP ativado." if config.use_amp else "AMP desativado por estabilidade numerica.")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("AVISO: CUDA nao disponivel. O treino vai rodar em CPU.")

    print("=" * 60 + "\n")
    return device


def maybe_show_or_save(path: Path, *, show_plots: bool, save_plots: bool) -> None:
    if save_plots:
        plt.savefig(path, dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()


def visualize_frequency(df, config: TrainConfig) -> None:
    print("\n[INFO] Generating Frequency Plot...")
    plt.figure(figsize=(10, 6))
    counts = df["cell_type"].value_counts()
    plt.barh(counts.index, counts.values)
    plt.title("Images in each class")
    plt.xlabel("Frequency")
    plt.ylabel("Class")
    plt.tight_layout()
    maybe_show_or_save(
        config.plots_dir / "frequency_plot.png",
        show_plots=config.show_plots,
        save_plots=config.save_plots,
    )


def visualize_samples(df, config: TrainConfig) -> None:
    print("\n[INFO] Generating Sample Images Plot...")
    unique_labels = df["cell_type"].unique()
    plt.figure(figsize=(15, 10))
    for index, label in enumerate(unique_labels):
        sample_path = df[df["cell_type"] == label].sample(1, random_state=config.seed)["path"].values[0]
        img = plt.imread(sample_path)
        plt.subplot(3, 3, index + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    maybe_show_or_save(
        config.plots_dir / "sample_images.png",
        show_plots=config.show_plots,
        save_plots=config.save_plots,
    )


def plot_confusion_matrix(cm_matrix, classes, config: TrainConfig, normalize=False, title="Confusion matrix") -> None:
    if normalize:
        row_sums = cm_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_matrix = cm_matrix.astype("float") / row_sums
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.figure(figsize=(8, 8))
    plt.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = np.nanmax(cm_matrix) / 2.0
    for i, j in itertools.product(range(cm_matrix.shape[0]), range(cm_matrix.shape[1])):
        value = cm_matrix[i, j]
        if np.isnan(value):
            value = 0.0
        plt.text(
            j,
            i,
            format(value, fmt),
            horizontalalignment="center",
            color="white" if value > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    maybe_show_or_save(
        config.plots_dir / "confusion_matrix.png",
        show_plots=config.show_plots,
        save_plots=config.save_plots,
    )


def plot_training_history(history, config: TrainConfig) -> None:
    if not history["epoch"]:
        return

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train")
    plt.plot(history["epoch"], history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train")
    plt.plot(history["epoch"], history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    maybe_show_or_save(
        config.plots_dir / "training_history.png",
        show_plots=config.show_plots,
        save_plots=config.save_plots,
    )


def denormalize_image(tensor, config: TrainConfig):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array(config.imagenet_mean)
    std = np.array(config.imagenet_std)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def generate_grad_cam(model, image_tensor, device, class_names, config: TrainConfig, true_label=None) -> None:
    model.eval()
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    target_layer = model.features[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(input_tensor)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        pred_idx = int(output.argmax(dim=1).item())
        pred_score = float(torch.softmax(output, dim=1)[0, pred_idx].item())

        model.zero_grad(set_to_none=True)
        output[:, pred_idx].backward()

        activation = activations[0][0]
        gradient = gradients[0][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activation).sum(dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (config.img_width, config.img_height))

        base_image = denormalize_image(image_tensor, config)
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.6 * base_image + 0.4 * heatmap, 0, 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(base_image)
        plt.title("Imagem")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap="jet")
        plt.title("Grad-CAM")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        title = f"Pred: {class_names[pred_idx]} ({pred_score:.2%})"
        if true_label is not None:
            title += f"\nTrue: {class_names[int(true_label)]}"
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        maybe_show_or_save(
            config.plots_dir / "grad_cam_example.png",
            show_plots=config.show_plots,
            save_plots=config.save_plots,
        )
    finally:
        forward_handle.remove()
        backward_handle.remove()
