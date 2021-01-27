import os
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score

from models.inception_resnet_v1 import InceptionResnetV1
from data.label_names import LabelNames


class L2DistanceClassifier:
    def __init__(self, model, number_persons, number_pictures) -> None:
        super().__init__()
        self.face_database = {}
        self.to_tensor = torchvision.transforms.ToTensor()
        self.threshold = 0.2
        self.number_persons = number_persons
        self.number_pictures_pp = number_pictures
        self.model = model
        self.model.eval()

        #TODO - edit FilePath
        self.datapath = "../datasets/data/"
        self.labeler = LabelNames("../data/data.p")

        print("Starting initaliziation for " + str(self.number_persons) + " Persons with " + str(self.number_pictures_pp) + " Images each")
        self.initalize()
        print("Finished initaliziation")
        print(" ")

    def initalize(self):
        folders = self.listdir_nohidden(self.datapath)
        for i in range(0, min(self.number_persons, len(folders))):
            label = self.labeler.read_from_pickle(folders[i])
            pictures = self.listdir_nohidden(os.path.join(self.datapath, folders[i]))
            if len(pictures)<(self.number_pictures_pp+10):
                print("-- Skipped ", label)
                pass
            else:
                for j in range(0, min(self.number_pictures_pp, round(len(pictures)))):
                    image = Image.open(self.datapath + folders[i] + "/" + pictures[j])
                    self.initalize_persons_by_img(image, label)
                print("-- Initalized ", label)

    def get_accuracy(self, threshold):
        y_true = []
        y_pred = []
        self.threshold = threshold
        folders = self.listdir_nohidden(self.datapath)
        for i in range(0, min(self.number_persons, len(folders))):
            label_true = self.labeler.read_from_pickle(folders[i])
            pictures = self.listdir_nohidden(os.path.join(self.datapath, folders[i]))
            for j in range(self.number_pictures_pp, min(self.number_pictures_pp+10, round(len(pictures)+10))):
                if len(pictures) < (self.number_pictures_pp + 10):
                    pass
                else:
                    image = Image.open(self.datapath + folders[i] + "/" + pictures[j])
                    label, distance = self.classify_by_img(image)
                    y_true.append(label_true)
                    y_pred.append(label)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def listdir_nohidden(self, path):
        return [f for f in os.listdir(path) if not f.startswith(".")]

    def img_path_to_encoding(self, image_path, model):
        img = cv2.imread(image_path, 1)
        return self.img_to_encoding(img, model)

    def img_to_encoding(self, image, model):
        image_tensor = self.to_tensor(image)
        image_tensor = image_tensor.unsqueeze(0)
        embedding = model(image_tensor)
        return embedding

    def get_min_dist(self, encoding, face_database):
        min_dist = 100
        for (name, encoded_image_name) in face_database.items():
            dist = torch.dist(encoding, encoded_image_name)
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist < self.threshold:
            return identity, min_dist
        else:
            return "Unknown", min_dist

    def make_aligned(self, inputName):
        detector = dlib.get_frontal_face_detector()
        # detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
        predictor = dlib.shape_predictor("data/shape_predictor_5_face_landmarks.dat")
        # img = dlib.load_rgb_image(inputName)
        img = np.array(inputName)
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
        path = "classifier/PeopleKnown/" + label
        if not os.path.exists(path):
            os.makedirs(path)
        output = Path(image_path).stem
        image_aligned = self.make_aligned(image_path)
        outputName = "classifier/PeopleKnown/" + label + "/" + output + ".jpeg"
        image_aligned.save(
            outputName, "JPEG", quality=80, optimize=True, progressive=True
        )
        # Save Identity in face_database
        identity = label
        self.face_database[identity] = self.img_to_encoding(image_aligned, self.model)
        return "Added Person"

    def initalize_persons_by_img(self, image, label: str):
        identity = label
        self.face_database[identity] = self.img_to_encoding(image, self.model)

    def classify(self, image_path):
        image_aligned = image_path
        identity, min_distance = self.get_min_dist(
            self.img_to_encoding(image_aligned, self.model), self.face_database
        )
        return identity, min_distance

    def classify_by_img(self, image):
        identity, min_distance = self.get_min_dist(
            self.img_to_encoding(image, self.model), self.face_database
        )
        return identity, min_distance

