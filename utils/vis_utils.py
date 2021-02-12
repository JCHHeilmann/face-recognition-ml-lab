from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from tqdm import tqdm


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
    model.eval()
    with torch.no_grad():
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
