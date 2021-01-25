import glob
import os
from os.path import join

import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset


class WebfaceDataset(Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

        self.image_filenames = self.read_file_paths()
        self.labels = []
        self.read_labels()
        #self.labels = self.read_labels()  # TODO JH fix

    def read_file_paths(self):
        return glob.glob(join(self.dataset_folder, "**/*.jpg"), recursive=True)

    # def read_labels(self):
    #     return [dir[0] for dir in os.walk(self.dataset_folder)]

    def read_labels(self):
        for filename in self.image_filenames:
            split_path = filename.split(os.sep)
            label = split_path[-2]
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

        return image_tensor, int(label)
