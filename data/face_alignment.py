from os.path import join

import dlib
import numpy as np
from PIL import Image


class FaceAlignment:
    def __init__(self):
        self.detector = dlib.cnn_face_detection_model_v1(
            "data/mmod_human_face_detector.dat"
        )
        self.predictor = dlib.shape_predictor(
            "data/shape_predictor_5_face_landmarks.dat"
        )

    def make_align(self, img):
        img = img.convert("RGB")
        height, width = img.size
        if width <= height:
            new_width = 128
            new_height = new_width * height // width
        else:
            new_height = 128
            new_width = new_height * width // height
        img = img.resize((new_height, new_width))

        img_array = np.array(img)

        detections = [det.rect for det in self.detector(img_array, 1)]

        num_faces = len(detections)

        if num_faces != 0:
            face = dlib.full_object_detections()
            face.append(self.predictor(img_array, detections[0]))

            image = dlib.get_face_chips(img_array, face)
            output_image = Image.fromarray(image[0]).convert("RGB").resize((128, 128))

            return output_image

        else:
            return None
