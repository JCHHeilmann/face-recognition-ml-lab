import dlib
from os.path import join
from PIL import Image

class FaceAlignment:

    def __init__(self):

        self.detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        self.predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        self.face = dlib.full_object_detections()

    def make_align(self,img):

        detections = [det.rect for det in self.detector(img, 1)]

        self.num_faces = len(detections)

        if self.num_faces != 0:

            self.face.append(self.predictor(img, detections[0]))

            image = dlib.get_face_chips(img, self.face, size=256)
            output_image = Image.fromarray(image[0])

            return output_image

        else:
            return None

