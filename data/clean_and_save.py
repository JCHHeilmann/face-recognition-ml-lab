from os.path import join
import glob
import FaceAlignment
from PIL import Image
import os

from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

data_folder = "../../../data/"
not_detected_file_path = join(data_folder, "not_detected.csv")


def process_folder(target_folder):
    split_path = target_folder.split("/")
    cln_name = join(data_folder, "Clean", split_path[-1])
    os.makedirs(cln_name)

    image_glob = join(data_folder, "CASIA-WebFace", split_path[-1], "*.jpg")
    target_files = glob.glob(image_glob)

    for target_file in target_files:

        img = Image.open(target_file)
        img = img.resize((256, 256))
        FaceAlignment.make_align(img, target_file, cln_name, not_detected_file_path)


if __name__ == "__main__":
    target_folders = glob.glob(join(data_folder, "CASIA-WebFace", "*"))

    process_map(
        process_folder,
        [folder for folder in target_folders],
        max_workers=mp.cpu_count(),
        chunksize=10,
    )
