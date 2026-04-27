from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from skin_lesion_classifier import TrainConfig, run_training_pipeline


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
PROJECT_DATA = KAGGLE_WORKING / "ham10000_data"
IMAGE_DIR = PROJECT_DATA / "all_images"


def copy_or_extract_file(source: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name

    if source.suffix.lower() == ".zip":
        print(f"Extracting {source}...")
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(destination_dir)
        return

    if destination.exists():
        return
    print(f"Copying {source.name}...")
    shutil.copy2(source, destination)


def prepare_kaggle_data() -> Path:
    if not KAGGLE_INPUT.exists():
        raise RuntimeError("This launcher is intended to run inside Kaggle.")

    PROJECT_DATA.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    metadata_candidates = list(KAGGLE_INPUT.rglob("HAM10000_metadata.csv")) + list(
        KAGGLE_INPUT.rglob("HAM10000_metadata")
    )
    if not metadata_candidates:
        raise FileNotFoundError(
            "HAM10000 metadata was not found. Add a Kaggle dataset containing HAM10000_metadata.csv."
        )

    metadata_source = metadata_candidates[0]
    metadata_destination = PROJECT_DATA / "HAM10000_metadata.csv"
    if not metadata_destination.exists():
        print(f"Copying metadata from {metadata_source}...")
        shutil.copy2(metadata_source, metadata_destination)

    image_files = [
        path
        for path in KAGGLE_INPUT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if image_files:
        print(f"Copying {len(image_files)} image files from Kaggle input...")
        for image_path in image_files:
            destination = IMAGE_DIR / image_path.name
            if not destination.exists():
                shutil.copy2(image_path, destination)

    image_zips = sorted(KAGGLE_INPUT.rglob("HAM10000_images_part*.zip"))
    for image_zip in image_zips:
        copy_or_extract_file(image_zip, IMAGE_DIR)

    if not any(IMAGE_DIR.glob("*.jpg")) and not any(IMAGE_DIR.glob("*.png")):
        raise FileNotFoundError(
            "No image files were prepared. Add HAM10000 image files or HAM10000_images_part*.zip to Kaggle input."
        )

    return PROJECT_DATA


def main() -> None:
    base_path = prepare_kaggle_data()
    config = TrainConfig(
        base_path=base_path,
        batch_size=16,
        gradient_accumulation_steps=1,
        num_workers=2,
        use_amp=True,
        show_plots=False,
        save_plots=True,
        run_grad_cam=True,
    )
    run_training_pipeline(config)

    checkpoint = config.best_model_path
    if checkpoint.exists():
        output_checkpoint = KAGGLE_WORKING / checkpoint.name
        shutil.copy2(checkpoint, output_checkpoint)
        print(f"Checkpoint copied to Kaggle output: {output_checkpoint}")

    if config.plots_dir.exists():
        output_plots = KAGGLE_WORKING / "plots"
        if output_plots.exists():
            shutil.rmtree(output_plots)
        shutil.copytree(config.plots_dir, output_plots)
        print(f"Plots copied to Kaggle output: {output_plots}")


if __name__ == "__main__":
    main()
