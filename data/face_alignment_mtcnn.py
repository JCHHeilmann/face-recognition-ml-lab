from os.path import join

import torch
from facenet_pytorch import MTCNN
from PIL import Image


class FaceAlignmentMTCNN:
    def __init__(self):
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            selection_method="probability",
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )

    def make_align(self, img):
        img = img.resize((512, 512))

        try:
            face = self.mtcnn(img)
            return face
        except:
            print("No Face")
            return None
