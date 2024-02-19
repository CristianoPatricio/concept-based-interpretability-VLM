import torch
from torch.utils.data import DataLoader
import clip

import itertools
import numpy as np
import pandas as pd

from modules.config import CFG
from modules.model import CLIPModel
from modules.engine import train_epoch, valid_epoch
from modules.data_setup import create_train_dataloader, create_val_dataloader
from modules.utils import set_seeds, extract_text_embeddings, extract_reference_embeddings, get_logger, \
    extract_image_embeddings, save_model


def extract_embeddings(cfg, model):
    # Extract text embeddings
    text_embeddings_clip = extract_text_embeddings(config=cfg, model=model)

    # Extract reference embeddings
    reference_embeddings_clip = extract_reference_embeddings(config=cfg, model=model)

    # Extract image embeddings
    image_embeddings_clip = extract_image_embeddings(config=cfg, model=model, preprocess=preprocess)

    return text_embeddings_clip, reference_embeddings_clip, image_embeddings_clip


def create_dataloaders(cfg, text_embeddings_clip, image_embeddings_clip):
    # Train DataLoader
    train_dataloader = create_train_dataloader(config=cfg, class_embeddings=text_embeddings_clip,
                                               images=image_embeddings_clip)

    # Val DataLoader
    val_dataloader = create_val_dataloader(config=cfg, class_embeddings=text_embeddings_clip,
                                           images=image_embeddings_clip)

    return train_dataloader, val_dataloader


def train(config):
    # Set seed for reproducibility
    set_seeds(cfg.seed)

    model = CLIPModel().to(config.device)
    params = [
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": config.head_lr, "weight_decay": config.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.patience, factor=config.factor
    )
    step = "epoch"

    best_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, val_dataloader)

        log.info(f"Epoch: {epoch + 1} - Train Loss: {train_loss.avg} - Val Loss: {valid_loss.avg}")

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            save_model(config, model)
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

    return model


if __name__ == '__main__':
    log = get_logger(__name__)  # init logger

    # Define configuration
    cfg = CFG

    log.info(f"Using {cfg.device}")
    log.info(f"Seed: {cfg.seed}")

    # Load CLIP model
    log.info(f"Loading CLIP model ({cfg.clip_model})...")
    model, preprocess = clip.load(cfg.clip_model, device=cfg.device)

    # Extract embeddings
    text_embeddings_clip, reference_embeddings_clip, image_embeddings_clip = extract_embeddings(cfg=cfg, model=model)

    # Creating DataLoaders
    train_dataloader, val_dataloader = create_dataloaders(cfg=cfg,
                                                          text_embeddings_clip=text_embeddings_clip,
                                                          image_embeddings_clip=image_embeddings_clip)

    # Train loop
    fine_tuned_model = train(config=cfg)
