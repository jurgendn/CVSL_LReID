from typing import Tuple

import numpy as np
import torch

device = torch.device("cuda")


def evaluate(gallery: dict, query: dict) -> Tuple[torch.Tensor, float]:
    """
    Evaluates image retrieval performance using Cumulative Matching Characteristics (CMC)
    and Mean Average Precision (mAP).

    Args:
        gallery (dict): A dictionary containing gallery image features, camera IDs, and labels.
        query (dict): A dictionary containing query image features, camera ID, and label.

    Returns:
        tuple[torch.Tensor, float]: A tuple containing the average CMC curve and mean Average Precision.
    """

    query_feature: torch.Tensor = query["feature"].to(device)
    query_cam: np.ndarray = np.array(query["camera"])
    query_label: np.ndarray = np.array(query["label"])
    gallery_feature: torch.Tensor = gallery["feature"].to(device)
    gallery_cam: np.ndarray = np.array(gallery["camera"])
    gallery_label: np.ndarray = np.array(gallery["label"])

    # print(query_feature.shape)
    cmc: torch.Tensor = torch.zeros(len(gallery_label), dtype=torch.int)
    ap: float = 0.0
    for i, _ in enumerate(query_label):
        ap_tmp, cmc_tmp = _evaluate(
            query_feature[i],
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam,
        )
        if cmc_tmp[0] == -1:
            continue
        cmc += cmc_tmp
        ap += ap_tmp
        # print(i, cmc_tmp[0])

    cmc = cmc.float()
    cmc /= len(query_label)  # average CMC

    # print(len(cmc))
    # print('-- Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
    return cmc, ap / len(query_label)


def _evaluate(
    query_feature: torch.Tensor,
    query_label: int,
    query_camera: int,
    gallery_feature: torch.Tensor,
    gallery_label: np.ndarray,
    gallery_camera: np.ndarray,
) -> Tuple[float, torch.Tensor]:
    """
    Evaluates a single query image against a gallery set using cosine similarity.

    Args:
        query_feature (torch.Tensor): The feature vector of the query image.
        query_label (int): The label of the query image.
        query_camera (int): The camera ID of the query image.
        gallery_feature (torch.Tensor): A tensor containing feature vectors of all gallery images.
        gallery_label (np.ndarray): A numpy array containing labels for all gallery images.
        gallery_camera (np.ndarray): A numpy array containing camera IDs for all gallery images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors containing the mAP and the CMC curve for the single query.
    """

    query = query_feature.unsqueeze(1)  # Reshape for matrix multiplication
    similarity_score = torch.mm(gallery_feature, query)
    similarity_score = similarity_score.squeeze(1).cpu().numpy()

    # Sort indices by descending similarity score
    sorted_indices = np.argsort(similarity_score)[::-1]

    # Identify relevant and irrelevant gallery images
    same_label_indices = np.where(gallery_label == query_label)[0]
    same_camera_indices = np.where(gallery_camera == query_camera)[0]
    relevant_indices = np.setdiff1d(
        same_label_indices, same_camera_indices, assume_unique=True
    )
    irrelevant_indices = np.concatenate(
        [np.where(gallery_label == -1)[0], same_camera_indices]
    )

    ap, cmc = compute_map(sorted_indices, relevant_indices, irrelevant_indices)
    return ap, cmc


def evaluate2(gallery: dict, query: dict) -> tuple[torch.Tensor, float]:
    """
    Evaluates image retrieval performance using Cumulative Matching Characteristics (CMC)
    and Mean Average Precision (mAP), considering clothing information for matching.

    Args:
        gallery (dict): A dictionary containing gallery image features, camera IDs, labels, and clothing types.
        query (dict): A dictionary containing query image features, camera ID, label, and clothing type.

    Returns:
        tuple[torch.Tensor, float]: A tuple containing the average CMC curve and mean Average Precision.
    """

    query_feature: torch.Tensor = query["feature"]
    query_camera: np.ndarray = np.array(query["camera"])
    query_label: np.ndarray = np.array(query["label"])
    query_cloth: np.ndarray = np.array(query["cloth"])
    gallery_feature: torch.Tensor = gallery["feature"]
    gallery_camera: np.ndarray = np.array(gallery["camera"])
    gallery_label: np.ndarray = np.array(gallery["label"])
    gallery_cloth: np.ndarray = np.array(gallery["cloth"])

    # print(query_feature.shape)
    cmc: torch.Tensor = torch.zeros(len(gallery_label), dtype=torch.int)
    ap: float = 0.0
    for i, _ in enumerate(query_label):
        ap_tmp, cmc_tmp = _evaluate2(
            query_feature[i],
            query_label[i],
            query_camera[i],
            query_cloth[i],
            gallery_feature,
            gallery_label,
            gallery_camera,
            gallery_cloth,
        )
        if cmc_tmp[0] == -1:
            continue
        cmc += cmc_tmp
        ap += ap_tmp
        # print(i, cmc_tmp[0])

    cmc = cmc.float()
    cmc /= len(query_label)  # average CMC

    # print(len(cmc))
    # print('-- Rank@1: %f, Rank@5: %f, Rank@10: %f, mAP: %f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
    return cmc, ap / len(query_label)


def _evaluate2(
    query_feature: torch.Tensor,
    query_label: int,
    query_camera: int,
    query_cloth: int,
    gallery_feature: torch.Tensor,
    gallery_label: np.ndarray,
    gallery_camera: np.ndarray,
    gallery_cloth: np.ndarray,
) -> Tuple[float, torch.Tensor]:
    """
    Evaluates a single query image against a gallery set using cosine similarity,
    considering clothing information for matching.

    Args:
        query_feature (torch.Tensor): The feature vector of the query image.
        query_label (int): The label of the query image.
        query_camera (int): The camera ID of the query image.
        query_cloth (int): The clothing type of the query image.
        gallery_feature (torch.Tensor): A tensor containing feature vectors of all gallery images.
        gallery_label (np.ndarray): A numpy array containing labels for all gallery images.
        gallery_camera (np.ndarray): A numpy array containing camera IDs for all gallery images.
        gallery_cloth (np.ndarray): A numpy array containing clothing types for all gallery images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors containing the mAP and the CMC curve for the single query.
    """

    query = query_feature.unsqueeze(1)  # Reshape for matrix multiplication
    similarity_score = torch.mm(gallery_feature, query)
    similarity_score = similarity_score.squeeze(1).cpu().numpy()

    # Sort indices by descending similarity score
    sorted_indices = np.argsort(similarity_score)[::-1]

    # Identify relevant and irrelevant gallery images based on label, camera, and clothing
    same_label_indices = np.where(gallery_label == query_label)[0]
    same_camera_indices = np.where(gallery_camera == query_camera)[0]
    same_cloth_indices = np.where(gallery_cloth == query_cloth)[0]
    relevant_indices = np.setdiff1d(
        np.setdiff1d(same_label_indices, same_camera_indices, assume_unique=True),
        same_cloth_indices,
        assume_unique=True,
    )
    irrelevant_indices = np.concatenate(
        [np.where(gallery_label == -1)[0], same_camera_indices, same_cloth_indices]
    )

    ap, cmc = compute_map(sorted_indices, relevant_indices, irrelevant_indices)
    return ap, cmc


def compute_map(
    sorted_indices: np.ndarray,
    relevant_indices: np.ndarray,
    irrelevant_indices: np.ndarray,
) -> tuple[float, torch.Tensor]:
    """
    Computes the mean Average Precision (mAP) and Cumulative Matching Characteristic (CMC) curve.

    Args:
        sorted_indices (np.ndarray): A 1D array of indices sorted by descending similarity scores.
        relevant_indices (np.ndarray): A 1D array containing indices of relevant gallery images.
        irrelevant_indices (np.ndarray): A 1D array containing indices of irrelevant gallery images.

    Returns:
        tuple[float, torch.Tensor]: A tuple containing the mAP value and the CMC curve as a torch.Tensor.
    """

    ap: float = 0.0
    cmc: torch.Tensor = torch.zeros(len(sorted_indices), dtype=torch.int)

    if len(relevant_indices) == 0:  # Handle empty relevant set
        cmc[0] = -1
        return ap, cmc

    # Filter out irrelevant indices
    mask = ~np.isin(sorted_indices, irrelevant_indices)
    sorted_indices = sorted_indices[mask]

    # Find positions of relevant indices within the sorted indices
    relevant_positions = np.searchsorted(sorted_indices, relevant_indices)

    # Calculate CMC curve and AP
    cmc[relevant_positions[0] :] = 1
    ngood = len(relevant_indices)
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (relevant_positions[i] + 1)
        old_precision = (
            i * 1.0 / relevant_positions[i] if relevant_positions[i] > 0 else 1.0
        )
        ap += d_recall * (old_precision + precision) / 2

    return ap, torch.tensor(cmc)
