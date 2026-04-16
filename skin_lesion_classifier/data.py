from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import TrainConfig
from .utils import ensure_dirs


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


def prepare_data(config: TrainConfig):
    print("\n[1] PREPARING DATA...")
    ensure_dirs(config)

    metadata_csv = config.base_path / "HAM10000_metadata.csv"
    metadata_raw = config.base_path / "HAM10000_metadata"
    if not metadata_raw.exists() and not metadata_csv.exists():
        archive_path = config.base_path / config.main_zip_name
        if archive_path.exists():
            print(f"Extracting {config.main_zip_name}...")
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(config.base_path)
        else:
            raise FileNotFoundError(f"File {config.main_zip_name} not found in {config.base_path}.")

    existing_images = len(list(config.image_dir.iterdir()))
    if existing_images < 10000:
        for zip_file in sorted(config.base_path.glob("HAM10000_images_part*.zip")):
            print(f"Unzipping {zip_file.name}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(config.image_dir)

    csv_path = metadata_csv if metadata_csv.exists() else metadata_raw
    df = pd.read_csv(csv_path)

    image_files = [path.name for path in config.image_dir.iterdir() if path.is_file()]
    if not image_files:
        raise FileNotFoundError(f"No images were found in {config.image_dir}.")

    ext = Path(image_files[0]).suffix
    df["path"] = df["image_id"].map(lambda image_id: str(config.image_dir / f"{image_id}{ext}"))
    df["cell_type"] = df["dx"].map(config.lesion_type_dict.get)
    df = df.dropna(subset=["cell_type"]).copy()
    df = df[df["path"].map(lambda value: Path(value).exists())].copy()

    class_to_idx = {class_name: index for index, class_name in enumerate(config.class_names)}
    df["label"] = df["cell_type"].map(class_to_idx)

    print(f"Data ready: {len(df)} images found.")
    return df, class_to_idx


def create_transforms(config: TrainConfig):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((config.img_height, config.img_width), scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.img_height, config.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(train_df, val_df, test_df, device, config: TrainConfig):
    print("\n[5] SETTING UP DATALOADERS...")
    train_transform, eval_transform = create_transforms(config)

    train_dataset = SkinCancerDataset(train_df, train_transform)
    val_dataset = SkinCancerDataset(val_df, eval_transform)
    test_dataset = SkinCancerDataset(test_df, eval_transform)

    use_cuda = device.type == "cuda"
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": use_cuda,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
