from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .config import TrainConfig


def get_model_builder(config: TrainConfig):
    if config.model_name == "efficientnet_b0":
        return models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT
    if config.model_name == "efficientnet_b2":
        return models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT
    raise ValueError(f"Unsupported MODEL_NAME: {config.model_name}")


def build_model(device, config: TrainConfig):
    print(f"\n[6] BUILDING {config.model_name.upper()} MODEL...")
    model_builder, default_weights = get_model_builder(config)

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
        nn.Linear(256, len(config.class_names)),
    )
    return model.to(device)


def freeze_feature_extractor(model) -> None:
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model, blocks_to_unfreeze: int) -> None:
    freeze_feature_extractor(model)
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-blocks_to_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True


def build_class_weights(train_df, device, config: TrainConfig):
    counts = train_df["label"].value_counts().sort_index()
    weights = 1.0 / np.sqrt(counts.values.astype(np.float32))
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_criterion(train_df, device, config: TrainConfig):
    if config.use_class_weights:
        class_weights = build_class_weights(train_df, device, config)
    else:
        class_weights = None
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
