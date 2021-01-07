from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from tqdm import tqdm


def imshow(inp, title=None, denormalize=False):
    """
    Reshape a tensor of image data to a grid for easy visualization.

    Inputs:
    - inp: Data of shape (C X H X W)
    - title: Default None
    - denormalize: indicator for inversing transformation, False by default
    """

    # reorder axes to move the channel dimension (H X W X C) instead of (C X H X W)
    inp = inp.numpy().transpose((1, 2, 0))

    # multiply by std and add the mean (i.e. inverse the transform)
    # Mean and Std will be updated based on transform function
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if denormalize:
        inp = std * inp + mean

        # make sure all numbers are between 0 and 1
        inp = np.clip(inp, 0, 1)

    # plot the image and add the title
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

    def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
        """
        Visualize the output of the prediction.
        By default this shows a batch of 10 images
        """

        #######################################################################
        # Implement later
        #######################################################################

        pass


def plot_and_log_embeddings(train_loader, model, xlim=None, ylim=None):
    train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
    plt = plot_embeddings(train_embeddings_otl, train_labels_otl, xlim, ylim)
    return plt


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    if torch.cuda.is_available():
        embeddings = embeddings.cpu()
        targets = targets.cpu()

    data = pd.DataFrame(
        {"dim_1": embeddings[:, 0], "dim_2": embeddings[:, 1], "targets": targets}
    )
    data["targets"] = data["targets"].astype(
        str
    )  # convert to string, so plotly interprets it as categorical variable

    fig = px.scatter(data, x="dim_1", y="dim_2", color="targets")
    return fig


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 512))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in tqdm(dataloader, total=len(dataloader)):
            if torch.cuda.is_available():
                images = images.cuda()
            embeddings[k : k + len(images)] = model(images).data.cpu().numpy()
            labels[k : k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels
