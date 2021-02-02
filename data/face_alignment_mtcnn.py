from os.path import join

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms


class FaceAlignmentMTCNN:
    def __init__(self):
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            selection_method="probability",
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )

        self.to_tensor = transforms.ToTensor()

    def make_align(self, img):
        img = img.resize((512, 512))

        try:
            bbx, prob = self.mtcnn.detect(img)
            if bbx is not None:
                self.mtcnn.extract(img, bbx, "temp.jpg")
                face = Image.open("temp.jpg")
                face_tensor = self.to_tensor(face)
                return face_tensor
            else:
                print("No Face")
                return None
        except:
            print("No Face")
            return None
