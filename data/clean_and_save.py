from os.path import join
import glob
from face_alignment import face_alignment
import os
import dlib

import multiprocessing as mp

data_folder = "../../../data/"
not_detected_file_path = join(data_folder, "not_detected.csv")

class clean_and_save:

    def process_file(self, target_file_path, cln_name):

        img = dlib.load_rgb_image(target_file_path)
        # aln_obj will be an image if face is detected, otherwise None.
        face_align_object = face_alignment()
        aln_obj = face_align_object.make_align(img)

        # If no face is detected, then saving filepath in csv file
        if aln_obj is None:
            with open(self.not_detected_file_path, "a") as csv_file_name:
                csv_file_name.write(target_file_path + ",\n")

        # Upon face detection, saving the cleaned image in JPEG format
        else:
            target_split = target_file_path.split("/")
            crnt_img = target_split[-1]
            aln_obj.save(join(cln_name, crnt_img), "JPEG")


    def process_folder(self, target_folder):
        split_path = target_folder.split("/")
        cln_name = join(self.data_folder, "Aligned_CASIA_WebFace", split_path[-1])
        os.makedirs(cln_name)

        image_glob = join(self.data_folder, "CASIA-WebFace", split_path[-1], "*.jpg")
        target_files = glob.glob(image_glob)

        for target_file in target_files:
            self.process_file(target_file,cln_name)

    def process_single_file(self,target_file_path, cln_name):
        target_file_path = target_file_path[0]
        self.process_file(target_file_path, cln_name)

if __name__ == "__main__":
    target_folders = glob.glob(join(data_folder, "CASIA-WebFace", "*"))

    with mp.Pool(processes=45) as pool:
        pool.map(clean_and_save.process_folder, [folder for folder in target_folders])
