from modules.data_setup import create_dataloader_inference
from modules.utils import (get_logger, get_image_embeddings, get_text_embeddings, extract_text_embeddings,
                           extract_reference_embeddings, extract_custom_text_embeddings, extract_image_embeddings, save_embeddings, extract_reference_embeddings)
from modules.config import CFG

import torch
from modules.model import CLIPModel
import clip


def extract_embeddings(cfg, model):
    # Extract text embeddings
    text_embeddings_clip = extract_text_embeddings(config=cfg, model=model)

    # Extract reference embeddings
    reference_embeddings_clip = extract_reference_embeddings(config=cfg, model=model)

    # Extract image embeddings
    image_embeddings_clip = extract_image_embeddings(config=cfg, model=model, preprocess=preprocess)

    return text_embeddings_clip, reference_embeddings_clip, image_embeddings_clip


if __name__ == '__main__':
    log = get_logger(__name__)  # init logger

    # Define configuration
    cfg = CFG

    log.info(f"Using {cfg.device}")
    log.info(f"Seed: {cfg.seed}")

    # Load CLIP model
    log.info(f"Loading CLIP model ({cfg.clip_model})...")
    model, preprocess = clip.load(cfg.clip_model, device=cfg.device)

    # Extract default embeddings
    text_embeddings_clip, reference_embeddings_clip, image_embeddings_clip = extract_embeddings(cfg=cfg, model=model)

    # Create DataLoader
    inference_dataloader = create_dataloader_inference(config=cfg, class_embeddings=text_embeddings_clip,
                                                       images=image_embeddings_clip)

    # Extract text embeddings from original CLIP
    class_labels_embeddings_clip = extract_custom_text_embeddings(config=cfg, model=model,
                                                                  type_of_text=(cfg.CLASS_LABELS_PROMPTS if cfg.dataset == "derm7pt" else cfg.CLASS_LABELS_PROMPTS_ISIC_2018))
    concepts_embeddings_clip = extract_custom_text_embeddings(config=cfg, model=model,
                                                              type_of_text=cfg.CONCEPT_PROMPTS)
    detailed_concepts_embeddings_clip = extract_custom_text_embeddings(config=cfg, model=model,
                                                                       type_of_text=cfg.DETAILED_CONCEPT_PROMPTS)

    # Extract image embeddings from Fine-Tuned CLIP
    _, test_image_embeddings = get_image_embeddings(config=cfg, test_dataloader=inference_dataloader,
                                                    model_path=cfg.path_to_model)
    save_embeddings(config=cfg, embeddings=test_image_embeddings, type="image_embeddings",
                    type_emb="image_embeddings", disease="")

    # Extract reference embeddings from Fine-Tuned CLIP
    _, test_reference_embeddings = get_text_embeddings(config=cfg, caption=reference_embeddings_clip,
                                                             model_path=cfg.path_to_model)
    save_embeddings(config=cfg, embeddings=test_reference_embeddings, type="text_embeddings",
                    type_emb="reference_embeddings", disease="")

    # Extract class label embeddings from Fine-Tuned CLIP
    for l in class_labels_embeddings_clip.keys():
        _, test_class_label_embeddings = get_text_embeddings(config=cfg, caption=class_labels_embeddings_clip[l],
                                                             model_path=cfg.path_to_model)
        save_embeddings(config=cfg, embeddings=test_class_label_embeddings, type="text_embeddings", type_emb="class_label_embeddings", disease=l)

    # Extract concept embeddings from Fine-Tuned CLIP
    for l in concepts_embeddings_clip.keys():
        _, test_concept_embeddings = get_text_embeddings(config=cfg, caption=concepts_embeddings_clip[l],
                                                         model_path=cfg.path_to_model)
        save_embeddings(config=cfg, embeddings=test_concept_embeddings, type="text_embeddings",
                        type_emb="concept_embeddings", disease=l)

    # Extract detailed concept embeddings from Fine-Tuned CLIP
    for l in detailed_concepts_embeddings_clip.keys():
        _, test_detailed_concept_embeddings = get_text_embeddings(config=cfg, caption=detailed_concepts_embeddings_clip[l],
                                                         model_path=cfg.path_to_model)
        save_embeddings(config=cfg, embeddings=test_detailed_concept_embeddings, type="text_embeddings",
                        type_emb="detailed_concept_embeddings", disease=l)

