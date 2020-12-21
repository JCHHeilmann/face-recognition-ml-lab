import glob
import os
from os.path import join

import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset


class WebfaceDataset(Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

        self.read_file_paths()
        self.encode_classes()

    def read_file_paths(self):
        self.image_filenames = glob.glob(
            join(self.dataset_folder, "**/*.jpg"), recursive=True
        )

    def encode_classes(self):
        self.class_to_idx = dict()
        self.labels = []

        for filename in self.image_filenames:
            split_path = filename.split(os.sep)
            label = split_path[-2]
            self.class_to_idx[label] = self.class_to_idx.get(
                label, len(self.class_to_idx)
            )
            self.labels.append(label)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx]).convert("RGB")
        image = image.resize((128, 128))
        split_path = self.image_filenames[idx].split(os.sep)
        label = split_path[-2]
        to_tensor = torchvision.transforms.ToTensor()

        image_tensor = to_tensor(image)

        return image_tensor, self.class_to_idx[label]

    def split(self):
        n_samples = len(self)
        shuffled_indices = np.random.permutation(n_samples)

        valandtest_ratio = 0.2
        testandvalidationset_inds = shuffled_indices[
            : int(n_samples * valandtest_ratio)
        ]
        trainingset_inds = shuffled_indices[int(n_samples * valandtest_ratio) :]

        test_ratio = 0.5
        n_samples_testandval = len(testandvalidationset_inds)
        validationset_inds = testandvalidationset_inds[
            : int(n_samples_testandval * test_ratio)
        ]
        testset_inds = testandvalidationset_inds[
            int(n_samples_testandval * test_ratio) :
        ]

        train_dataset = torch.utils.data.Subset(self, indices=trainingset_inds)
        val_dataset = torch.utils.data.Subset(self, indices=validationset_inds)
        test_dataset = torch.utils.data.Subset(self, indices=testset_inds)

        return train_dataset, val_dataset, test_dataset
