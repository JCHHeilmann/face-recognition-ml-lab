import torch
import torchvision
from models.inception_resnet_v1 import InceptionResnetV1
import os
import numpy as np
import dlib
from PIL import Image
import cv2
from pathlib import Path


to_tensor = torchvision.transforms.ToTensor()

#load latest checkpoint
checkpoint = torch.load("charmed-cosmos-135_epoch_19", map_location=torch.device("cpu"))
model = InceptionResnetV1()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

#Calculate the Embeddings from one Image with Path
def img_path_to_encoding(image_path, model):
    img = cv2.imread(image_path, 1)
    return img_to_encoding(img, model)

#Calculate the Embeddings from one Image
def img_to_encoding(image, model):
    image_tensor = to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)
    embedding = model(image_tensor)
    return embedding


#make and initalize face_database of known faces
face_database = {}

def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def initalize():
    listdir_nohidden("./PeopleKnown")
    PathKnownPersons = './PeopleKnown'
    for name in listdir_nohidden(PathKnownPersons):
        for image in listdir_nohidden(os.path.join(PathKnownPersons,name)):
            identity = os.path.splitext(os.path.basename(image))[0]
            face_database[identity] = img_path_to_encoding(os.path.join(PathKnownPersons,name,image),model)

#Calculate the smallest distance between embedding and face_database
def get_min_dist(encoding, face_database):
    min_dist = 100
    for(name, encoded_image_name) in face_database.items():
        dist = np.linalg.norm(encoding.detach() - encoded_image_name.detach())
        if(dist < min_dist):
            min_dist = dist
            identity = name
    if min_dist < threshold:
        print("Face :", identity, "--- Distance :", min_dist)
    else:
        print("Face :", "No Matches", "--- Distance :", min_dist)

#set the threshold
threshold = 0.2


#make the aligned images
#TODO: needs to be changed so that we use the one from Dave
def make_aligned(inputName):
    detector = dlib.get_frontal_face_detector()
    #detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
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
    #img.save(inputName, "JPEG", quality=80, optimize=True, progressive=True)
    return img


#Function to classify images
def classify_image(image_path):
    #makes them aligned
    image_aligned = make_aligned(image_path)
    #image_aligned.show()
    get_min_dist(img_to_encoding(image_aligned, model), face_database)


#Function to add the Images to the Known Faces
def add_face_to_known(image_path, name):
    path = "./PeopleKnown/" + name
    if not os.path.exists(path):
        os.makedirs(path)
    output = Path(image_path).stem
    image_aligned = make_aligned(image_path)
    outputName = "./PeopleKnown/" + name + "/" + output + ".jpeg"
    image_aligned.save(outputName, "JPEG", quality=80, optimize=True, progressive=True)

    # Save Identity in face_database
    identity = name
    face_database[identity] = img_to_encoding(image_aligned, model)
