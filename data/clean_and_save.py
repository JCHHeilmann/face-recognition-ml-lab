from os.path import join
import glob
import FaceAlignment
from PIL import Image
import os
from tqdm import tqdm

txt_file_name = "not_detected.txt"
target_folders = glob.glob(join("../../../data/CASIA-WebFace", "*"))
for target_folder in tqdm(enumerate(target_folders), total=len(target_folders)):
    split_path = target_folder.split("/")
    cln_name = join("Clean", split_path[-1])
    os.makedirs(cln_name)

    target_files = glob.glob(join("CASIA-WebFace", split_path[-1], "*.jpg"))

    for target_file in target_files:

        img = Image.open(target_file)
        img = img.resize((256, 256))
        FaceAlignment.make_align(img, target_file, cln_name)

