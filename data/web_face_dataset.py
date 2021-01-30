import glob
import os
from os.path import join

import numpy as np
import torchvision.transforms
from facenet_pytorch import fixed_image_standardization
from PIL import Image
from torch.utils.data import Dataset


class WebfaceDataset(Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.to_tensor = torchvision.transforms.ToTensor()
        self.image_filenames = self.read_file_paths()
        self.labels = self.get_labels()

        ###Test with fixed_image_standardization
        self.transform = torchvision.transforms.Compose(
            [np.float32, torchvision.transforms.ToTensor(), fixed_image_standardization]
        )

        ###Test with more transforms
        self.transform_new = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.RandomResizedCrop(160),
                torchvision.transforms.RandomHorizontalFlip(),
                np.float32,
                torchvision.transforms.ToTensor(),
                fixed_image_standardization,
            ]
        )

    def read_file_paths(self):
        paths = glob.glob(join(self.dataset_folder, "*/*.png"), recursive=True)
        if len(paths) == 0:
            paths = glob.glob(join(self.dataset_folder, "*/*.jpg"), recursive=True)
        return paths

    def get_labels(self):
        return [path.split("/")[-2] for path in self.image_filenames]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx]).convert("RGB")

        # image = image.resize((160, 160))
        split_path = self.image_filenames[idx].split(os.sep)
        label = split_path[-2]

        image_tensor = self.to_tensor(image)
        # image_tensor = self.transform_new(image)
        #image_tensor = self.transform(image)

        return image_tensor, int(label)
