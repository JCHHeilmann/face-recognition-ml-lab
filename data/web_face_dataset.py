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
        self.labels = self.get_labels()

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
        image = image.resize((128, 128))
        split_path = self.image_filenames[idx].split(os.sep)
        label = split_path[-2]
        to_tensor = torchvision.transforms.ToTensor()

        image_tensor = to_tensor(image)

        return image_tensor, int(label)
