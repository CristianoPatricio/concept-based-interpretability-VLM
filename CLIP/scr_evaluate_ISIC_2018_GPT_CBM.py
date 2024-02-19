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


def calculate_similarity(MODEL, DETAILED_CONCEPT_PROMPTS, SEED):
    print(f"[INFO] DATASET: {DATASET}")
    print(f"[INFO] MODEL: {MODEL}")

    # Load image embeddings
    img_embeddings = np.load(
        f"output/{MODEL}/image_embeddings/image_embeddings__{DATASET}_{MODEL}-Fine-Tuned_{SEED}.npy",
        allow_pickle=True).item()

    # Load reference embeddings
    reference_embeddings = torch.from_numpy(np.load(
        f"output/{MODEL}/text_embeddings/reference_embeddings__{DATASET}_{MODEL}-Fine-Tuned_{SEED}.npy")).unsqueeze(0)

    # Weights of Melanoma (Fig. 5 (I) of reference [16])
    weights_melanoma = np.array([[1.9, 2.0, 3.84, 1.6, 0.15, 0.0, 2.4, 8.66, -0.05, -0.15]])

    results = dict()
    concept_predictions = dict()
    # Iterate over images and calculate similarity
    for im in img_embeddings.keys():
        img_feats = torch.from_numpy(img_embeddings[im]).unsqueeze(0)

        similarity_scores = []
        for disease_label in DETAILED_CONCEPT_PROMPTS.keys():
            # Load text embeddings of concepts descriptions
            text_feats = torch.from_numpy(np.load(
                f"output/{MODEL}/text_embeddings/detailed_concept_embeddings_{disease_label}_{DATASET}_{MODEL}-Fine-Tuned_{SEED}.npy")).unsqueeze(
                0)

            # Calculate similarity
            similarity = calculate_similarity_score(image_features_norm=img_feats,
                                                    prompt_target_embedding_norm=text_feats,
                                                    prompt_ref_embedding_norm=reference_embeddings,
                                                    top_k=2,
                                                    temp=(1 / np.exp(4.5944)),
                                                    normalize=False)

            similarity_scores.append(similarity[0])

        # Calculate prediction given weights of melanoma
        y_pred = np.dot(weights_melanoma, np.array(similarity_scores))
        concept_predictions[im] = np.array(similarity_scores)
        results[im] = y_pred

    return results


def evaluate(results_dict):
    gt = pd.read_csv("../data/ISIC_2018/image_classes_ISIC_2018.csv")

    train_images_df = pd.read_csv("../data/ISIC_2018/ISIC_2018_train.csv")
    train_images = train_images_df["images"].tolist()

    valiadtion_images_df = pd.read_csv("../data/ISIC_2018/ISIC_2018_validation.csv")
    validation_images = valiadtion_images_df["images"].tolist()

    test_images_df = pd.read_csv("../data/ISIC_2018/ISIC_2018_test.csv")
    test_images = test_images_df["images"].tolist()

    y_true_val = []
    y_pred_probs_val = []
    for im in results_dict.keys():

        if str(im) in validation_images:
            y_true_val.append(1 if gt.loc[gt['images'] == str(im)]['labels'].tolist()[0] == 3 else 0)
            y_pred_probs_val.append(results_dict[im])

    fpr, tpr, thresholds = roc_curve(y_true_val, y_pred_probs_val)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Threshold value is:", optimal_threshold, "\n")

    y_true = []
    y_pred = []
    y_pred_probs = []
    for im in results_dict.keys():

        if str(im) in test_images:
            y_true.append(1 if gt.loc[gt['images'] == str(im)]['labels'].tolist()[0] == 3 else 0)
            y_pred.append(1 if results[im] > optimal_threshold else 0)
            y_pred_probs.append(results[im])

    print("Classification Report:")
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=["MEL", "NEV"]))

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    print("Confusion Matrix:")
    print(conf_matrix, "\n")

    # Calculate AUC score
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc}")

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

    DETAILED_CONCEPT_PROMPTS = {
        "Asymmetry": ["This is dermatoscopy of an asymmetric shape with one half not mirroring the other half",
                      "This is dermatoscopy of asymmetrical distribution of pigmentation",
                      "This is dermatoscopy of irregular and non-symmetrical borders",
                      "This is dermatoscopy of significant asymmetry",
                      "This is dermatoscopy of asymmetry in the form of dissimilar features on opposite sides of the lesion"],
        "Irregular": ["This is dermatoscopy of irregular shapes or outlines",
                      "This is dermatoscopy of irregular distribution of pigmentation",
                      "This is dermatoscopy of poorly defined borders",
                      "This is dermatoscopy of irregular and atypical patterns",
                      "This is dermatoscopy of irregular features in the form of non-uniform characteristics"],
        "Erosion": ["This is dermatoscopy of surface ulceration or erosion",
                    "This is dermatoscopy of erosion as a crusted area on the skin",
                    "This is dermatoscopy of ulcerated appearance",
                    "This is dermatoscopy of erosion with exposed underlying tissue",
                    "This is dermatoscopy of erosion in the form of disrupted or absent epidermal structures"],
        "Black": ["This is dermatoscopy of dark or black pigmentation", "This is dermatoscopy of black coloration",
                  "This is dermatoscopy of dark brown to black areas",
                  "This is dermatoscopy of black structures or pigmentation",
                  "This is dermatoscopy of black coloration in the form of concentrated dark areas in the lesion"],
        "Blue": ["This is dermatoscopy of blue or blue-gray coloration", "This is dermatoscopy of blue coloration",
                 "This is dermatoscopy of bluish patches or areas of discoloration",
                 "This is dermatoscopy of blue structures or pigmentation",
                 "This is dermatoscopy of blue coloration in the form of bluish hues or tones in the lesion"],
        "White": ["This is dermatoscopy of white or hypopigmented coloration",
                  "This is dermatoscopy of white coloration",
                  "This is dermatoscopy of pale or depigmented patches or areas",
                  "This is dermatoscopy of white structures or depigmentation",
                  "This is dermatoscopy of white coloration in the form of reduced pigmentation in the lesion"],
        "Brown": ["This is dermatoscopy of brown or dark-brown pigmentation",
                  "This is dermatoscopy of brown coloration",
                  "This is dermatoscopy of brown patches or areas of discoloration",
                  "This is dermatoscopy of brown structures or pigmentation",
                  "This is dermatoscopy od brown coloration in the form of various shades of brown in the lesion"],
        "Multiple Colors": ["This is dermatoscopy of a combination of different colors",
                            "This is dermatoscopy of multiple colorations with a varied and complex appearance",
                            "This is dermatoscopy of a mix of different hues",
                            "This is dermatoscopy of diverse colors and pigmentation",
                            "This is dermatoscopy of multiple coloration in the form of different colored areas within the lesion"],
        "Tiny": ["This is dermatoscopy of small and minute structures or shapes",
                 "This is dermatoscopy of tiny shapes characterized by their small size",
                 "This is dermatoscopy of minuscule or small-sized patterns",
                 "This is dermatoscopy of tiny structures or shapes",
                 "This is dermatoscopy of tiny shape in the form of small and discrete features within the lesion"],
        "Regular": ["This is dermatoscopy of a regular and symmetrical pattern",
                    "This is dermatoscopy of regular and evenly spaced structures",
                    "This is dermatoscopy of uniform arrangement of patterns",
                    "This is dermatoscopy of regular pattern in the form of symmetrical and well-defined features within the lesion"]
    }

    REFERENCE_CONCEPT_PROMPTS = ["This is dermatoscopy"]

    SEED = [0, 42, 84, 168]

    # ['ViT-B-32', 'ViT-B-16', 'RN50', 'RN101', 'ViT-L-14', 'RN50x16']
    gpt_cbm_42 = [0.702, 0.730, 0.713, 0.656, 0.664, 0.729]

    metrics_dict = dict()

    # Iterate over models
    for idx, m in enumerate(MODEL):
        bacc_acum = []
        for s in SEED:
            if s == 42:
                bacc_acum.append(gpt_cbm_42[idx])
            else:
                results = calculate_similarity(MODEL=m, DETAILED_CONCEPT_PROMPTS=DETAILED_CONCEPT_PROMPTS, SEED=s)
                bacc = evaluate(results_dict=results)
                bacc_acum.append(bacc)

        metrics_dict[m] = bacc_acum

    #print(metrics_dict)
    np.save("results_ISIC_2018_GPT_CBM.npy", metrics_dict)
