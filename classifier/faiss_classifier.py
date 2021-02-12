import os
from random import randint

import numpy as np
import torch
import torchvision
from facenet_pytorch import MTCNN
from numpy.lib.twodim_base import mask_indices

from data.face_alignment_mtcnn import FaceAlignmentMTCNN
from data.label_names import LabelNames
from models.inception_resnet_v1 import InceptionResnetV1

if os.uname().machine == "ppc64le":
    import faiss_ppc as faiss
else:
    import faiss


class FaissClassifier:
    def __init__(
        self,
        index="datasets/vector_generous_jazz_2021-02-02_11-53-36.index",
        model=None,
    ) -> None:
        super().__init__()
        self.mtcnn = MTCNN(image_size=160, margin=0, selection_method="probability")
        self.threshold = 0.001
        self.to_tensor = torchvision.transforms.ToTensor()
        self.indexIDMap = faiss.read_index(index)
        self.dictionary = LabelNames("data/data.p")

        if os.uname().machine != "ppc64le":
            self.preprocessor = FaceAlignmentMTCNN()

        if not model:
            self.checkpoint = torch.load(
                "checkpoints/generous-jazz-275_epoch_19",
                map_location=torch.device("cpu"),
            )
            self.model = InceptionResnetV1()
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        else:
            self.model = model

        self.model.eval()

    def img_tensor_to_encoding(self, image_tensor, model):
        image_tensor = image_tensor.unsqueeze(0)
        embedding = model(image_tensor).data.cpu().numpy()
        return embedding

    def img_to_encoding(self, image, model):
        image_tensor = self.to_tensor(image)
        image_tensor = image_tensor.unsqueeze(0)
        embedding = model(image_tensor).data.cpu().numpy()
        return embedding

    def random_n_digits(self, n):
        range_start = 10 ** (n - 1)
        range_end = (10 ** n) - 1
        return randint(range_start, range_end)

    def classify(self, embeddings):
        k = 1
        distance, label = self.indexIDMap.search(embeddings.astype("float32"), k)
        if distance < self.threshold:
            return int(label)
        else:
            return 0

        # alternative: use a vote of the closest 10 to determine the classification result
        k = 10
        distances, labels = self.indexIDMap.search(embeddings.astype("float32"), k)

        distance_mask = np.array([d < self.threshold for d in distances])
        valid_labels = labels[distance_mask]

        if len(valid_labels) > 0:
            _, indices, counts = np.unique(
                valid_labels, return_counts=True, return_index=True
            )

            maximums = np.argwhere(counts == np.max(counts))
            candidates = indices[maximums]
            result_index = np.min(candidates)
            return int(valid_labels[result_index])
        else:
            return 0

    def classify_with_surroundings(self, image):
        image_aligned_tensor = self.preprocessor.make_align(image)  # MTCNN aligment

        if image_aligned_tensor == None:
            print("No face found")
            return ["Unknown"], None

        embedding = self.img_tensor_to_encoding(
            image_aligned_tensor, self.model
        )  # MTCNN aligment

        k = 100
        distances, labels, embeddings = self.indexIDMap.search_and_reconstruct(
            embedding.astype("float32"), k
        )

        if distances[0][0] < self.threshold:
            label_names = [
                self.dictionary.read_from_pickle(label) for label in labels[0]
            ]

            label_names = [label_names[0] + " "] + label_names
            embeddings = embeddings[0].tolist()
            embeddings = embedding.tolist() + embeddings

            return label_names, embeddings
        else:
            return ["Unknown"], None

        # alternative: use a vote of the closest 10 to determine the classification result
        labels = labels[0]
        distances = distances[0]
        embeddings = embeddings[0]

        distance_mask = [d < self.threshold for d in distances]
        valid_labels = labels[distance_mask]

        if len(valid_labels) > 0:
            _, indices, counts = np.unique(
                valid_labels, return_counts=True, return_index=True
            )
            maximums = np.argwhere(counts == np.max(counts))
            candidates = indices[maximums]
            result_index = np.min(candidates)
            result = valid_labels[result_index]

            result_index = np.where(labels == result)[0]
            temp = labels[0]
            labels[0] = labels[result_index]
            labels[result_index] = temp

            temp = embeddings[0]
            embeddings[0] = embeddings[result_index]
            embeddings[result_index] = temp

            label_names = [self.dictionary.read_from_pickle(label) for label in labels]

            label_names = [label_names[0] + " "] + label_names
            embeddings = embeddings.tolist()
            embeddings = embedding.tolist() + embeddings

            return label_names, embeddings
        else:
            return ["Unknown"], None

    def add_person(self, image, name: str):
        image_align_tensor = self.preprocessor.make_align(image)

        embedding_new = self.img_tensor_to_encoding(image_align_tensor, self.model)

        random_label = self.random_n_digits(8)
        random_label_array = np.array([random_label])

        self.indexIDMap.add_with_ids(embedding_new, random_label_array)
        self.dictionary.add_name(name, random_label)
