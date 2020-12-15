import matplotlib.pyplot as plt
import numpy as np
import torch

# Adopted from the PyTorch transfer learning turorial


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


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(
            embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i]
        )
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    return plt


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 512))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            embeddings[k : k + len(images)] = (
                model.get_embedding(images).data.cpu().numpy()
            )
            labels[k : k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels
