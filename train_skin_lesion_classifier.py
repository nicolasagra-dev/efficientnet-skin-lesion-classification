import glob
import itertools
import os
import random
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ================= CONFIGURATION =================
MAIN_ZIP_NAME = "dataverse_files.zip"
BASE_PATH = os.getcwd()
IMAGE_DIR = os.path.join(BASE_PATH, "all_images")
SAVE_DIR = os.path.join(BASE_PATH, "Saved_Model")
PLOTS_DIR = os.path.join(BASE_PATH, "plots")

# Perfil focado em melhorar accuracy na GTX 1650 4 GB
MODEL_NAME = "efficientnet_b2"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 20
IMG_HEIGHT = 260
IMG_WIDTH = 260
HEAD_LEARNING_RATE = 3e-4
FINETUNE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
UNFREEZE_LAST_BLOCKS = 3
SEED = 42
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 1e-4
USE_AMP = False
GRAD_CLIP_NORM = 1.0
USE_CLASS_WEIGHTS = False
LABEL_SMOOTHING = 0.05
TTA_PASSES = 4

SHOW_PLOTS = False
SAVE_PLOTS = True
RUN_GRAD_CAM = True

LESION_TYPE_DICT = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}
CLASS_NAMES = list(LESION_TYPE_DICT.values())
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device():
    print("=" * 60)
    print("CONFIGURANDO DISPOSITIVO...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detectada: {gpu_name}")
        print(f"Memoria total: {total_memory_gb:.2f} GB")
        print("PyTorch vai treinar usando CUDA.")
        print(f"Modelo: {MODEL_NAME}")
        print(f"Resolucao: {IMG_HEIGHT}x{IMG_WIDTH}")
        print(f"Batch real: {BATCH_SIZE}")
        print(f"Acumulacao de gradiente: {GRADIENT_ACCUMULATION_STEPS}")
        if USE_AMP:
            print("AMP ativado.")
        else:
            print("AMP desativado por estabilidade numerica.")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("AVISO: CUDA nao disponivel. O treino vai rodar em CPU.")

    print("=" * 60 + "\n")
    return device


def ensure_dirs():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    if SAVE_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)


def prepare_data():
    print("\n[1] PREPARING DATA...")
    ensure_dirs()

    if not os.path.exists("HAM10000_metadata") and not os.path.exists("HAM10000_metadata.csv"):
        if os.path.exists(MAIN_ZIP_NAME):
            print(f"Extracting {MAIN_ZIP_NAME}...")
            with zipfile.ZipFile(MAIN_ZIP_NAME, "r") as zip_ref:
                zip_ref.extractall(BASE_PATH)
        else:
            raise FileNotFoundError(f"File {MAIN_ZIP_NAME} not found in {BASE_PATH}.")

    existing_images = len(os.listdir(IMAGE_DIR))
    if existing_images < 10000:
        internal_zips = glob.glob("HAM10000_images_part*.zip")
        for zip_file in internal_zips:
            print(f"Unzipping {zip_file}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(IMAGE_DIR)

    csv_path = "HAM10000_metadata.csv" if os.path.exists("HAM10000_metadata.csv") else "HAM10000_metadata"
    df = pd.read_csv(csv_path)

    image_files = [name for name in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, name))]
    if not image_files:
        raise FileNotFoundError(f"No images were found in {IMAGE_DIR}.")

    first_img = image_files[0]
    ext = os.path.splitext(first_img)[1]

    df["path"] = df["image_id"].map(lambda image_id: os.path.join(IMAGE_DIR, image_id + ext))
    df["cell_type"] = df["dx"].map(LESION_TYPE_DICT.get)
    df = df.dropna(subset=["cell_type"]).copy()
    df = df[df["path"].apply(os.path.exists)].copy()

    class_to_idx = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
    df["label"] = df["cell_type"].map(class_to_idx)

    print(f"Data ready: {len(df)} images found.")
    return df, class_to_idx


def maybe_show_or_save(path):
    if SAVE_PLOTS:
        plt.savefig(path, dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def visualize_frequency(df):
    print("\n[INFO] Generating Frequency Plot...")
    plt.figure(figsize=(10, 6))
    counts = df["cell_type"].value_counts()
    plt.barh(counts.index, counts.values)
    plt.title("Images in each class")
    plt.xlabel("Frequency")
    plt.ylabel("Class")
    plt.tight_layout()
    maybe_show_or_save(os.path.join(PLOTS_DIR, "frequency_plot.png"))


def visualize_samples(df):
    print("\n[INFO] Generating Sample Images Plot...")
    unique_labels = df["cell_type"].unique()
    plt.figure(figsize=(15, 10))
    for i, label in enumerate(unique_labels):
        sample_path = df[df["cell_type"] == label].sample(1, random_state=SEED)["path"].values[0]
        img = plt.imread(sample_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    maybe_show_or_save(os.path.join(PLOTS_DIR, "sample_images.png"))


def plot_confusion_matrix(cm_matrix, classes, normalize=False, title="Confusion matrix"):
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
    maybe_show_or_save(os.path.join(PLOTS_DIR, "confusion_matrix.png"))


class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, label


def create_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(train_df, val_df, test_df, device):
    print("\n[5] SETTING UP DATALOADERS...")
    train_transform, eval_transform = create_transforms()

    train_dataset = SkinCancerDataset(train_df, train_transform)
    val_dataset = SkinCancerDataset(val_df, eval_transform)
    test_dataset = SkinCancerDataset(test_df, eval_transform)

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": use_cuda,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def get_model_builder():
    if MODEL_NAME == "efficientnet_b0":
        return models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT
    if MODEL_NAME == "efficientnet_b2":
        return models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT
    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")


def build_model(device):
    print(f"\n[6] BUILDING {MODEL_NAME.upper()} MODEL...")
    model_builder, default_weights = get_model_builder()

    try:
        model = model_builder(weights=default_weights)
        print("Pesos pretrained do ImageNet carregados com sucesso.")
    except Exception as error:
        print(f"AVISO: nao foi possivel carregar pesos pretrained ({error}).")
        print("Continuando com pesos aleatorios.")
        model = model_builder(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(256, len(CLASS_NAMES)),
    )
    model = model.to(device)
    return model


def freeze_feature_extractor(model):
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model, blocks_to_unfreeze):
    freeze_feature_extractor(model)
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-blocks_to_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True


def build_class_weights(train_df, device):
    counts = train_df["label"].value_counts().sort_index()
    weights = 1.0 / np.sqrt(counts.values.astype(np.float32))
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_criterion(train_df, device):
    if USE_CLASS_WEIGHTS:
        class_weights = build_class_weights(train_df, device)
    else:
        class_weights = None
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)


def build_tta_variants(images, passes):
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


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, training=False, tta_passes=1):
    if training:
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
            with autocast(device_type=device.type, enabled=device.type == "cuda" and USE_AMP):
                logits = model(images)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                loss = criterion(logits, targets)

            if training:
                if not torch.isfinite(loss):
                    print("AVISO: batch ignorado por perda nao finita durante o treino.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss_to_backprop = loss / GRADIENT_ACCUMULATION_STEPS
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

                should_step = (
                    batch_index % GRADIENT_ACCUMULATION_STEPS == 0
                    or batch_index == len(loader)
                )
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                probs = torch.softmax(logits, dim=1)
            else:
                if tta_passes > 1:
                    probs = predict_probabilities(model, images, tta_passes=tta_passes)
                else:
                    probs = torch.softmax(logits, dim=1)

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


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, class_names, stage_name):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "image_size": (IMG_HEIGHT, IMG_WIDTH),
            "stage_name": stage_name,
            "model_name": MODEL_NAME,
        },
        path,
    )


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def denormalize_image(tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def generate_grad_cam(model, image_tensor, device, class_names, true_label=None):
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
        cam = cv2.resize(cam, (IMG_WIDTH, IMG_HEIGHT))

        base_image = denormalize_image(image_tensor)
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
        maybe_show_or_save(os.path.join(PLOTS_DIR, "grad_cam_example.png"))
    finally:
        forward_handle.remove()
        backward_handle.remove()


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
):
    optimizer = Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda" and USE_AMP)
    epochs_without_improvement = 0

    print(f"\n[7] STARTING {stage_name.upper()}...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
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

        if val_acc > best_val_acc + EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                history["epoch"][-1],
                best_val_acc,
                CLASS_NAMES,
                stage_name,
            )
            print(f"Best model saved to: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No meaningful improvement for {epochs_without_improvement} epoch(s). "
                f"Patience: {EARLY_STOPPING_PATIENCE}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping acionado em {stage_name}.")
            break

    return best_val_acc, history


def plot_training_history(history):
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
    maybe_show_or_save(os.path.join(PLOTS_DIR, "training_history.png"))


def main():
    set_seed(SEED)
    device = configure_device()
    df, class_to_idx = prepare_data()

    if SAVE_PLOTS or SHOW_PLOTS:
        visualize_frequency(df)
        visualize_samples(df)

    print("\n[4] SPLITTING DATA (TRAIN/VAL/TEST)...")
    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["cell_type"], random_state=SEED)
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["cell_type"],
        random_state=SEED,
    )

    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, device)
    model = build_model(device)
    criterion = create_criterion(train_df, device)

    best_model_path = os.path.join(SAVE_DIR, f"best_skin_lesion_{MODEL_NAME}_1650.pth")
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
        epochs=HEAD_EPOCHS,
        learning_rate=HEAD_LEARNING_RATE,
        best_model_path=best_model_path,
        best_val_acc=best_val_acc,
        history=history,
    )

    load_checkpoint(best_model_path, model, device)
    unfreeze_last_blocks(model, UNFREEZE_LAST_BLOCKS)
    best_val_acc, history = train_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        stage_name="fine_tuning",
        epochs=FINETUNE_EPOCHS,
        learning_rate=FINETUNE_LEARNING_RATE,
        best_model_path=best_model_path,
        best_val_acc=best_val_acc,
        history=history,
    )

    if SAVE_PLOTS or SHOW_PLOTS:
        plot_training_history(history)

    print("\n[8] EVALUATION...")
    checkpoint = load_checkpoint(best_model_path, model, device)
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    print(f"Best stage: {checkpoint.get('stage_name', 'unknown')}")

    test_loss, test_acc, test_probs, test_preds, test_targets = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        scaler=None,
        training=False,
        tta_passes=TTA_PASSES,
    )
    test_probs = np.nan_to_num(test_probs, nan=0.0, posinf=1.0, neginf=0.0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    target_names = [idx_to_class[index] for index in range(len(idx_to_class))]

    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names, zero_division=0))

    cm_matrix = confusion_matrix(test_targets, test_preds)
    plot_confusion_matrix(cm_matrix, target_names, normalize=True, title="Confusion Matrix")

    print("\nGenerating ROC curve data...")
    lb = LabelBinarizer()
    lb.fit(range(len(target_names)))
    y_test_bin = lb.transform(test_targets)

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
    maybe_show_or_save(os.path.join(PLOTS_DIR, "roc_curve.png"))

    if RUN_GRAD_CAM:
        print("\n[9] GENERATING GRAD-CAM EXAMPLE...")
        sample_images, sample_targets = next(iter(test_loader))
        generate_grad_cam(
            model,
            sample_images[0],
            device,
            target_names,
            true_label=sample_targets[0].item(),
        )
        print("Grad-CAM salvo em plots/grad_cam_example.png")
    else:
        print("\n[9] Grad-CAM desativado.")

    print("O treino e a inferencia usam a GTX 1650 via CUDA quando disponivel.")


if __name__ == "__main__":
    main()
