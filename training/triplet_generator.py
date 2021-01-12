from itertools import combinations

import numpy as np
import torch
import wandb


def pairwise_distances(vectors):
    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    distance_matrix = (
        -2 * vectors.mm(torch.t(vectors))
        + vectors.pow(2).sum(dim=1).view(1, -1)
        + vectors.pow(2).sum(dim=1).view(-1, 1)
    )
    return distance_matrix


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    if margin==0:
        semihard_negatives = hardest_negative(loss_values)
        return semihard_negatives
    else:
        semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
        return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def get_triplets(embeddings, labels, device, margin, negative_selection_fn):
    if device == "cuda":
        embeddings = embeddings.cuda()
    else:
        embeddings = embeddings.cpu()
    distance_matrix = pairwise_distances(embeddings)
    if device == "cuda":
        distance_matrix = distance_matrix.cuda()
    else:
        distance_matrix = distance_matrix.cpu()

    labels = labels.cpu().data.numpy()
    triplets = []

    number_of_anchors = 0
    for label in set(labels):
        label_mask = labels == label
        label_indices = np.where(label_mask)[0]
        # If no same labels in batch skip
        if len(label_indices) < 2:
            continue
        else:
            number_of_anchors += 1

        negative_indices = np.where(np.logical_not(label_mask))[
            0
        ]  # Find all indices of images with False label
        anchor_positives = list(
            combinations(label_indices, 2)
        )  # All anchor-positive pairs
        anchor_positives = np.array(anchor_positives)

        # anchor_positives[:, 0] - anchor, anchor_positives[:, 1] - positive
        anchor_positive_distances = distance_matrix[
            anchor_positives[:, 0], anchor_positives[:, 1]
        ]

        for anchor_positive, anchor_positive_distance in zip(
            anchor_positives, anchor_positive_distances
        ):
            # calculating loss based on anchor_postive and every possible negative embedding
            loss_values = (
                anchor_positive_distance
                - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])),
                    torch.LongTensor(negative_indices),
                ]
                + margin
            )
            loss_values = loss_values.data.cpu().numpy()

            # return indices of the image that have hard/semihard/random loss
            hard_negative = negative_selection_fn(loss_values)
            max_dist = np.argmax(loss_values)

            if hard_negative is not None:
                hard_negative = negative_indices[hard_negative]
                triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

    triplets = np.array(triplets)

    return torch.LongTensor(triplets)


def get_hardest(embeddings, labels, device, margin):
    return get_triplets(
        embeddings, labels, device, margin, negative_selection_fn=hardest_negative
    )


def get_random(embeddings, labels, device, margin):
    return get_triplets(
        embeddings, labels, device, margin, negative_selection_fn=random_hard_negative
    )


def get_semihard(embeddings, labels, device, margin):
    return get_triplets(
        embeddings,
        labels,
        device,
        margin,
        negative_selection_fn=lambda x: semihard_negative(x, margin),
    )
