import numpy as np
import dlib
from os.path import join
from PIL import Image
from imutils import face_utils


def get_eye_nose(shape):
    nose = shape[4]
    outer_left = shape[2]
    outer_right = shape[0]
    return nose, outer_left, outer_right


def get_crop_dim(nse, otr_lft, otr_rht):

    x_mid = (otr_rht[0] + otr_lft[0]) / 2
    y_mid = (otr_lft[1] + nse[1]) / 2

    crop_dim = (x_mid - 50, y_mid - 50, x_mid + 50, y_mid + 50)
    return crop_dim


def get_angle(otr_lft, otr_rht):

    theta = np.arctan((otr_rht[1] - otr_lft[1]) / (otr_rht[0] - otr_lft[0]))
    theta = np.degrees(theta)
    return theta


def get_extraction(src_img, nse, otr_lft, otr_rht):
    ext_img = Image.fromarray(src_img)
    crop_dim = get_crop_dim(nse, otr_lft, otr_rht)
    ext_img = ext_img.crop(crop_dim)
    return ext_img


def make_align(src_img, face_file_path, target_folder, not_detected_file_path):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    src_img = np.array(src_img)

    rects = detector(src_img, 0)
    if len(rects) == 0:

        # print("Sorry, there were no faces found in '{}'".format(face_file_path))
        with open(not_detected_file_path, "a") as txt_file_name:
            txt_file_name.write(face_file_path + ",")

    else:
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(src_img, rect)
            shape = face_utils.shape_to_np(shape)

        nse, otr_lft, otr_rht = get_eye_nose(shape)

        angle = get_angle(otr_lft, otr_rht)

        src_img = Image.fromarray(src_img)
        rot_img = np.array(src_img.rotate(angle))

        # re-recognition of points after rotation
        r_rects = detector(rot_img, 0)
        if len(r_rects) == 0:
            # print("Sorry, there were no faces found in '{}'".format(face_file_path))
            with open(not_detected_file_path, "a") as txt_file_name:
                txt_file_name.write(face_file_path + ",")

        else:
            for (r_i, r_rect) in enumerate(r_rects):
                r_shape = predictor(rot_img, r_rect)
                r_shape = face_utils.shape_to_np(r_shape)

            r_nse, r_otr_lft, r_otr_rht = get_eye_nose(r_shape)

            ext_img = get_extraction(rot_img, r_nse, r_otr_lft, r_otr_rht)
            ext_img = ext_img.resize((256, 256))

            target_split = face_file_path.split("/")
            crnt_img = target_split[-1]

            ext_img.save(join(target_folder, crnt_img), "JPEG")

