import torch


def get_data_loaders(
    dataset, batch_size, train_proportion, val_proportion, test_proportion
):
    dataset_length = len(dataset)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [
            int(train_proportion * dataset_length),
            int(val_proportion * dataset_length),
            int(test_proportion * dataset_length),
        ],
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
