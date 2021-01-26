import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, targets):

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        triplets = self.triplet_selector(embeddings, targets, device, self.margin)

        if len(triplets) == 0:
            return None, 0

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        anchor_positive_distances = (
            (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]])
            .pow(2)
            .sum(1)
            .pow(0.5)
        )  # .pow(.5)
        anchor_negative_distances = (
            (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]])
            .pow(2)
            .sum(1)
            .pow(0.5)
        )  # .pow(.5)
        losses = F.relu(
            anchor_positive_distances - anchor_negative_distances + self.margin
        )

        print(
            f"\n{(len([1 for loss in losses if loss == torch.tensor(0.0)]) / len(losses)) * 100} %\n"
        )
        return losses.mean(), len(triplets)
