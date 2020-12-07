import torch


def get_data_loaders(
    dataset, batch_size, train_proportion, val_proportion, test_proportion
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
