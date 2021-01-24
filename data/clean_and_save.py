import glob
import multiprocessing as mp
import os
from os.path import join

from PIL import Image

from .face_alignment import FaceAlignment

# data_folder = "../../../data/"
data_folder = "datasets/"
not_detected_file_path = join(data_folder, "not_detected.csv")


class CleanAndSave:
    def __init__(self):
        pass

    def process_file(self, target_file_path, cln_name):

        img = Image.open(target_file_path)
        # aln_obj will be an image if face is detected, otherwise None.
        aln_obj = self.face_align_object.make_align(img)

        # If no face is detected, then saving filepath in csv file
        if aln_obj is None:
            return
        #     self.csv_file.write(target_file_path + ",\n")

        # Upon face detection, saving the cleaned image in PNG format
        else:
            target_split = target_file_path.split("/")
            crnt_img = target_split[-1].split(".")[0] + ".png"
            aln_obj.save(join(cln_name, crnt_img))

    def process_folder(self, target_folder):
        self.face_align_object = FaceAlignment()

        split_path = target_folder.split("/")
        cln_name = join(data_folder, "Aligned_CASIA_WebFace", split_path[-1])
        os.makedirs(cln_name)

        image_glob = join(data_folder, "CASIA-WebFace", split_path[-1], "*.jpg")
        target_files = glob.glob(image_glob)

        for target_file in target_files:
            self.process_file(target_file, cln_name)

        # self.csv_file.close()

    def process_single_file(self, target_file_path, cln_name):
        target_file_path = target_file_path[0]
        self.process_file(target_file_path, cln_name)


if __name__ == "__main__":
    target_folders = glob.glob(join(data_folder, "CASIA-WebFace", "*"))
    clean_and_save = CleanAndSave()

    with mp.Pool(processes=45) as pool:
        pool.map(clean_and_save.process_folder, [folder for folder in target_folders])
