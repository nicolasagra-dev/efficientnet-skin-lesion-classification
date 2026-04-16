from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


LESION_TYPE_DICT = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    main_zip_name: str = "dataverse_files.zip"
    model_name: str = "efficientnet_b2"
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    head_epochs: int = 5
    finetune_epochs: int = 20
    img_height: int = 260
    img_width: int = 260
    head_learning_rate: float = 3e-4
    finetune_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    num_workers: int = 2
    unfreeze_last_blocks: int = 3
    seed: int = 42
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4
    use_amp: bool = False
    grad_clip_norm: float = 1.0
    use_class_weights: bool = False
    label_smoothing: float = 0.05
    tta_passes: int = 4
    show_plots: bool = False
    save_plots: bool = True
    run_grad_cam: bool = True
    lesion_type_dict: dict[str, str] = field(default_factory=lambda: dict(LESION_TYPE_DICT))
    base_path: Path = field(default_factory=Path.cwd)
    image_dir: Path = field(init=False)
    save_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    class_names: tuple[str, ...] = field(init=False)
    imagenet_mean: tuple[float, float, float] = field(default=IMAGENET_MEAN)
    imagenet_std: tuple[float, float, float] = field(default=IMAGENET_STD)

    def __post_init__(self) -> None:
        self.image_dir = self.base_path / "all_images"
        self.save_dir = self.base_path / "Saved_Model"
        self.plots_dir = self.base_path / "plots"
        self.class_names = tuple(self.lesion_type_dict.values())

    @property
    def best_model_path(self) -> Path:
        return self.save_dir / f"best_skin_lesion_{self.model_name}_1650.pth"
