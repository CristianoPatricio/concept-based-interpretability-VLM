import glob
from timeit import default_timer as timer

import torch
import clip
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.special
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, balanced_accuracy_score, auc

# Define agnostic device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_similarity_score(image_features_norm,
                               prompt_target_embedding_norm,
                               prompt_ref_embedding_norm,
                               temp=1,
                               top_k=-1,
                               normalize=True):
    """
    Similarity Score used in "Fostering transparent medical image AI via an image-text foundation model grounded in medical literature"
    https://www.medrxiv.org/content/10.1101/2023.06.07.23291119v1.full.pdf
    """

    target_similarity = prompt_target_embedding_norm.float() @ image_features_norm.T.float()
    ref_similarity = prompt_ref_embedding_norm.float() @ image_features_norm.T.float()

    if top_k > 0:
        idx_target = target_similarity.argsort(dim=1, descending=True)
        target_similarity_mean = target_similarity[:, idx_target.squeeze()[:top_k]].mean(dim=1)

        ref_similarity_mean = ref_similarity.mean(dim=1)
    else:
        target_similarity_mean = target_similarity.mean(dim=1)
        ref_similarity_mean = ref_similarity.mean(dim=1)

    if normalize:
        similarity_score = scipy.special.softmax([target_similarity_mean.numpy(), ref_similarity_mean.numpy()], axis=0)[
                           0, :].mean(axis=0)
    else:
        similarity_score = target_similarity_mean.mean(axis=0)

    return similarity_score


def calculate_similarity(MODEL, CLASS_LABELS_PROMPTS, SEED):
    print(f"[INFO] DATASET: {DATASET}")
    print(f"[INFO] MODEL: {MODEL}")

    # Load image embeddings
    img_embeddings = np.load(
        f"output/{MODEL}/image_embeddings/image_embeddings__{DATASET}_{MODEL}-Fine-Tuned_{SEED}.npy",
        allow_pickle=True).item()

    # Load reference embeddings
    reference_embeddings = torch.from_numpy(np.load(
        f"output/{MODEL}/text_embeddings/reference_embeddings__{DATASET}_{MODEL}-Fine-Tuned_{SEED}.npy")).unsqueeze(0)

    results = dict()
    # Iterate over images and calculate similarity
    for im in img_embeddings.keys():
        img_feats = torch.from_numpy(img_embeddings[im]).unsqueeze(0)

        similarity_scores = []
        for disease_label in CLASS_LABELS_PROMPTS.keys():
            # Load text embeddings
            text_feats = torch.from_numpy(np.load(
                f"output/{MODEL}/text_embeddings/class_label_embeddings_{disease_label}_{DATASET}_{MODEL}-Fine-Tuned_{SEED}.npy"))[0].unsqueeze(
                0)

            # Calculate similarity
            similarity = calculate_similarity_score(image_features_norm=img_feats,
                                                    prompt_target_embedding_norm=text_feats,
                                                    prompt_ref_embedding_norm=reference_embeddings,
                                                    top_k=-1,
                                                    temp=(1 / np.exp(4.5944)),
                                                    normalize=False)

            similarity_scores.append(similarity)

        # Save score into a dictionary w.r.t. to image
        results[im] = similarity_scores

    return results


def evaluate(results_dict):
    gt = pd.read_csv("../data/ISIC_2018/image_classes_ISIC_2018.csv")

    train_images_df = pd.read_csv("../data/ISIC_2018/ISIC_2018_train.csv")
    train_images = train_images_df["images"].tolist()

    valiadtion_images_df = pd.read_csv("../data/ISIC_2018/ISIC_2018_validation.csv")
    validation_images = valiadtion_images_df["images"].tolist()

    test_images_df = pd.read_csv("../data/ISIC_2018/ISIC_2018_test.csv")
    test_images = test_images_df["images"].tolist()

    y_true = []
    y_pred = []
    y_pred_probs = []
    for im in results_dict.keys():

        if str(im) in test_images:
            y_true.append(1 if gt.loc[gt['images'] == str(im)]['labels'].tolist()[0] == 3 else 0)
            y_pred.append(1 if np.argmax(results_dict[im]) == 3 else 0)
            y_pred_probs.append(results_dict[im])

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    print("Confusion Matrix:")
    print(conf_matrix, "\n")

    # BACC
    bacc = balanced_accuracy_score(y_true, y_pred)
    print(f"BACC: {bacc}")

    # Sensitivity
    SE = TP / (TP + FN)
    print(f"Sensitivity: {SE}")

    # Specificity
    SP = TN / (TN + FP)
    print(f"Specificity: {SP}")

    return bacc


if __name__ == '__main__':
    # CONSTANTS
    MODEL = ['ViT-B-32', 'ViT-B-16', 'RN50', 'RN101', 'ViT-L-14', 'RN50x16']
    DATASET = "ISIC_2018"
    ADDITIONAL_COMMENTS = "CLASS_LABELS_PROMPTS"

    CLASS_LABELS_PROMPTS = {
        "BKL": ["This is dermatoscopy of pigmented benign keratosis",
                'This is dermoscopy of pigmented benign keratosis'],
        "NV": ["This is dermatoscopy of nevus", 'This is dermoscopy of nevus'],
        "DF": ['This is dermatoscopy of dermatofibroma', 'This is dermoscopy of dermatofibroma'],
        "MEL": ['This is dermatoscopy of melanoma', 'This is dermoscopy of melanoma'],
        "VASC": ['This is dermatoscopy of vascular lesion', 'This is dermoscopy of vascular lesion'],
        "BCC": ['This is dermatoscopy of basal cell carcinoma', 'This is dermoscopy of basal cell carcinoma'],
        "AKIEC": ['This is dermatoscopy of actinic keratosis', 'This is dermoscopy of actinic keratosis']
    }

    REFERENCE_CONCEPT_PROMPTS = ["This is dermatoscopy", "This is dermoscopy"]

    SEED = [0, 42, 84, 168]

    metrics_dict = dict()

    # Iterate over models
    for idx, m in enumerate(MODEL):
        bacc_acum = []
        for s in SEED:
            results = calculate_similarity(MODEL=m, CLASS_LABELS_PROMPTS=CLASS_LABELS_PROMPTS, SEED=s)
	    bacc = evaluate(results_dict=results)
	    bacc_acum.append(bacc)

        metrics_dict[m] = bacc_acum

    np.save("results_ISIC_2018_baseline.npy", metrics_dict)
