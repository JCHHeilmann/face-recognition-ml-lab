from os.path import join
import glob
import face_alignment
from PIL import Image
import os

import multiprocessing as mp

# data_folder = "../../../data/"
data_folder = "../datasets/"
not_detected_file_path = join(data_folder, "not_detected.csv")


def process_folder(target_folder):
    split_path = target_folder.split("/")
    cln_name = join(data_folder, "Aligned_CASIA_WebFace", split_path[-1])
    os.makedirs(cln_name)

    image_glob = join(data_folder, "CASIA-WebFace", split_path[-1], "*.jpg")
    target_files = glob.glob(image_glob)

    for target_file in target_files:
        face_alignment.make_align(target_file, cln_name, not_detected_file_path)


if __name__ == "__main__":
    target_folders = glob.glob(join(data_folder, "CASIA-WebFace", "*"))

    with mp.Pool(processes=10) as pool:
        pool.map(process_folder, [folder for folder in target_folders])
