from os.path import join
import glob
import face_alignment
from PIL import Image
import os
import dlib

import multiprocessing as mp

data_folder = "../../../data/"
# data_folder = "../datasets/"
not_detected_file_path = join(data_folder, "not_detected.csv")


def process_folder(target_folder):
    split_path = target_folder.split("/")
    cln_name = join(data_folder, "Aligned_CASIA_WebFace", split_path[-1])
    os.makedirs(cln_name)

    image_glob = join(data_folder, "CASIA-WebFace", split_path[-1], "*.jpg")
    target_files = glob.glob(image_glob)

    for target_file in target_files:
        img = dlib.load_rgb_image(target_file)

        # aln_obj will be an image if face is detected, otherwise None.
        aln_obj = face_alignment.make_align(img)

        #If no face is detected, then saving filepath in csv file
        if aln_obj is None:
            with open(not_detected_file_path, "a") as csv_file_name:
                csv_file_name.write(target_file + ",\n")

        #Upon face detection, saving the cleaned image in JPEG format
        else:
            target_split = target_file.split("/")
            crnt_img = target_split[-1]
            aln_obj.save(join(cln_name, crnt_img), "JPEG")

if __name__ == "__main__":
    target_folders = glob.glob(join(data_folder, "CASIA-WebFace", "*"))

    with mp.Pool(processes=45) as pool:
        pool.map(process_folder, [folder for folder in target_folders])
