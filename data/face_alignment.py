import dlib
from os.path import join
from PIL import Image


def make_align(img):

    detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

    detections = [det.rect for det in detector(img, 1)]

    num_faces = len(detections)

    if num_faces != 0:
        face = dlib.full_object_detections()

        face.append(predictor(img, detections[0]))

        image = dlib.get_face_chips(img, face, size=256)
        output_image = Image.fromarray(image[0])

        return output_image

    else:
        return None
