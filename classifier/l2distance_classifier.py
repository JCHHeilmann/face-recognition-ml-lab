import os
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torchvision
from PIL import Image

from models.inception_resnet_v1 import InceptionResnetV1


class L2DistanceClassifier:
    def __init__(self) -> None:
        super().__init__()
        self.face_database = {}
        self.to_tensor = torchvision.transforms.ToTensor()
        self.threshold = 0.2

        self.checkpoint = torch.load(
            "charmed-cosmos-135_epoch_19", map_location=torch.device("cpu")
        )
        self.model = InceptionResnetV1()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        print("Starting initaliziation")
        self.initalize()
        print("Ended initaliziation")

    def initalize(self):
        self.listdir_nohidden("./PeopleKnown")
        PathKnownPersons = "./PeopleKnown"
        for name in self.listdir_nohidden(PathKnownPersons):
            # i=0
            for image in self.listdir_nohidden(os.path.join(PathKnownPersons, name)):
                identity = os.path.splitext(os.path.basename(image))[0]
                # identity = name + "_" + str(i)
                self.face_database[identity] = self.img_path_to_encoding(
                    os.path.join(PathKnownPersons, name, image), self.model
                )
                # i = i+1

    def listdir_nohidden(self, path):
        return [f for f in os.listdir(path) if not f.startswith(".")]

    def img_path_to_encoding(self, image_path, model):
        img = cv2.imread(image_path, 1)
        return self.img_to_encoding(img, model)

        # Calculate the Embeddings from one Image

    def img_to_encoding(self, image, model):
        image_tensor = self.to_tensor(image)
        image_tensor = image_tensor.unsqueeze(0)
        embedding = model(image_tensor)
        return embedding

    def get_min_dist(self, encoding, face_database):
        min_dist = 100
        for (name, encoded_image_name) in face_database.items():
            dist = np.linalg.norm(encoding.detach() - encoded_image_name.detach())
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist < self.threshold:
            return identity, min_dist
        else:
            return "No Match", min_dist

    def make_aligned(self, inputName):
        detector = dlib.get_frontal_face_detector()
        # detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        predictor = dlib.shape_predictor("../data/shape_predictor_5_face_landmarks.dat")
        img = dlib.load_rgb_image(inputName)
        dets = detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found")
            exit()
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(predictor(img, detection))
        image = dlib.get_face_chip(img, faces[0])
        img = Image.fromarray(image)
        # img.save(inputName, "JPEG", quality=80, optimize=True, progressive=True)
        return img

    def add_person(self, image_path, label: str):
        path = "./PeopleKnown/" + label
        if not os.path.exists(path):
            os.makedirs(path)
        output = Path(image_path).stem
        image_aligned = self.make_aligned(image_path)
        outputName = "./PeopleKnown/" + label + "/" + output + ".jpeg"
        image_aligned.save(
            outputName, "JPEG", quality=80, optimize=True, progressive=True
        )
        # Save Identity in face_database
        identity = label
        self.face_database[identity] = self.img_to_encoding(image_aligned, self.model)
        return "Added Person"

    def classify(self, image_path):
        # makes them aligned
        image_aligned = self.make_aligned(image_path)
        identity, min_distance = self.get_min_dist(
            self.img_to_encoding(image_aligned, self.model), self.face_database
        )
        return identity, min_distance
