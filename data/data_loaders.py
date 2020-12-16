from data.balanced_batch_sampler import BalancedBatchSampler
import torch
import numpy as np


def get_data_loaders(
    dataset,
    classes_per_batch,
    samples_per_class,
    train_proportion,
    val_proportion,
    test_proportion,
):
    dataset_size = len(dataset)

    train_size = int(train_proportion * dataset_size)
    val_size = int(val_proportion * dataset_size)
    test_size = int(test_proportion * dataset_size)

    train_size += dataset_size - (
        train_size + val_size + test_size
    )  # correct rounding errors with train set size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
    )

    train_labels = np.array(dataset.labels)[train_dataset.indices]
    balanced_sampler = BalancedBatchSampler(
        train_labels, classes_per_batch, samples_per_class
    )

    batch_size = classes_per_batch * samples_per_class

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=balanced_sampler
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
