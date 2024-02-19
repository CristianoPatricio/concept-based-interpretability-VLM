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

def calculate_similarity(DATASET, MODEL, CONCEPT_PROMPTS):
    print(f"[INFO] DATASET: {DATASET}")
    print(f"[INFO] MODEL: {MODEL}")

    # Load image embeddings
    img_embeddings = np.load(f"img_embeddings/image_embeddings_{DATASET}_MONET_ViT-L-14_Segmented_Norm.npy",
                             allow_pickle=True).item()

    # Load reference embeddings
    reference_embeddings = torch.from_numpy(
        np.load(f"reference_embeddings/reference_concept_embeddings.npy")).unsqueeze(0)

    # Weights of Melanoma (Fig. 5 (I) of reference [16])
    weights_melanoma = np.array([[1.9, 2.0, 3.84, 1.6, 0.15, 0.0, 2.4, 8.66, -0.05, -0.15]])

    results = dict()
    concept_predictions = dict()
    # Iterate over images and calculate similarity
    for im in img_embeddings.keys():
        img_feats = torch.from_numpy(img_embeddings[im]).unsqueeze(0)

        similarity_scores = []
        for concept in CONCEPT_PROMPTS.keys():
            # Load text embeddings of concepts
            text_feats = torch.from_numpy(np.load(f"text_embeddings/concept_embeddings_{concept}.npy")).unsqueeze(0)

            # Calculate similarity
            similarity = calculate_similarity_score(image_features_norm=img_feats,
                                                    prompt_target_embedding_norm=text_feats,
                                                    prompt_ref_embedding_norm=reference_embeddings,
                                                    top_k=-1,
                                                    temp=(1 / np.exp(4.5944)),
                                                    normalize=False)

            similarity_scores.append(similarity[0])

        # Calculate prediction given weights of melanoma
        y_pred = np.dot(weights_melanoma, np.array(similarity_scores))
        concept_predictions[im] = np.array(similarity_scores)
        results[im] = y_pred

    return results

def evaluate(results_dict, fold):
    # Evaluation
    gt = pd.read_csv("../data/PH2/PH2_dataset.csv")

    valiadtion_images_df = pd.read_csv(f"../data/PH2/PH2_train_split_{fold}.csv")
    validation_images = valiadtion_images_df["images"].tolist()

    test_images_df = pd.read_csv(f"../data/PH2/PH2_test_split_{fold}.csv")
    test_images = test_images_df["images"].tolist()

    y_true_val = []
    y_pred_probs_val = []
    for im in results_dict.keys():

        if str(im) in validation_images:
            y_true_val.append(gt.loc[gt['images'] == str(im)]['labels'].tolist()[0])
            y_pred_probs_val.append(np.max(results_dict[im]))

    fpr, tpr, thresholds = roc_curve(y_true_val, y_pred_probs_val)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Threshold value is:", optimal_threshold, "\n")

    y_true = []
    y_pred = []
    y_pred_probs = []
    for im in results_dict.keys():

        if str(im) in test_images:
            y_true.append(gt.loc[gt['images'] == str(im)]['labels'].tolist()[0])
            y_pred.append(1 if np.max(results_dict[im]) > optimal_threshold else 0)
            y_pred_probs.append(np.max(results_dict[im]))

    print(f"Classification Report:")
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=["NEV", "MEL"]))

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    print(f"Confusion Matrix:")
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
    MODEL= ['ViT-L-14']
    DATASET = "PH2"
    ADDITIONAL_COMMENTS = "CLASS_LABELS_PROMPTS"

    CONCEPT_PROMPTS = {
        "Asymmetry": ["This is dermatoscopy of an asymmetry"],
        "Irregular": ["This is dermatoscopy of irregular"],
        "Erosion": ["This is dermatoscopy of erosion"],
        "Black": ["This is dermatoscopy of black"],
        "Blue": ["This is dermatoscopy of blue"],
        "White": ["This is dermatoscopy of white"],
        "Brown": ["This is dermatoscopy of brown"],
        "Multiple Colors": ["This is dermatoscopy of multiple colors"],
        "Tiny": ["This is dermatoscopy of tiny"],
        "Regular": ["This is dermatoscopy of a regular"]
    }

    REFERENCE_CONCEPT_PROMPTS = ["This is dermatoscopy"]

    NO_FOLDS = 5

    metrics_dict = dict()

    # Iterate over models
    for idx, m in enumerate(MODEL):
        results = calculate_similarity(DATASET, m, CONCEPT_PROMPTS)

        bacc_acum = []
        for f in range(NO_FOLDS):
            bacc = evaluate(results_dict=results, fold=f)
            bacc_acum.append(bacc)

        metrics_dict[m] = bacc_acum

    np.save("results_PH2_CBM.npy", metrics_dict)




