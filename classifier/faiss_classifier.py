from random import randint

import faiss_ppc
import numpy as np
import torch
import torchvision
from data.face_alignment import FaceAlignment
from data.label_names import LabelNames
from models.inception_resnet_v1 import InceptionResnetV1


class FaissClassifier:
    def __init__(self) -> None:
        super().__init__()

        self.threshold = 0.00001
        self.to_tensor = torchvision.transforms.ToTensor()
        self.indexIDMap = faiss_ppc.read_index("./classifier/vector.index")
        self.dictionary = LabelNames("./data/data.p")
        self.preprocessor = FaceAlignment()

        self.checkpoint = torch.load(
            "checkpoints/charmed-cosmos-135_epoch_19", map_location=torch.device("cpu")
        )
        self.model = InceptionResnetV1()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    def img_to_encoding(self, image, model):
        image_tensor = self.to_tensor(image)
        image_tensor = image_tensor.unsqueeze(0)
        embedding = model(image_tensor).data.cpu().numpy()
        return embedding

    def random_n_digits(n):
        range_start = 10 ** (n - 1)
        range_end = (10 ** n) - 1
        return randint(range_start, range_end)

    def classify(self, image):

        image_aligned = self.preprocessor.make_align(image)
        embedding = self.img_to_encoding(image_aligned, self.model)

        k = 1
        distance, label = self.indexIDMap.search(embedding.astype("float32"), k)

        if distance < self.threshold:
            return self.dictionary.read_from_pickle(label)
        else:
            return "Unknown"

    def classify_with_surroundings(self, image):

        image_aligned = self.preprocessor.make_align(image)
        embedding = self.img_to_encoding(image_aligned, self.model)

        k = 50
        distance, label = self.indexIDMap.search(embedding.astype("float32"), k)

        return self.dictionary.read_from_pickle(label)

        # TODO: return also embeddings
        # return  # label_name, surrounding_embeddings

    def add_person(self, image, labels: str):

        image_align = self.preprocessor.make_align(image)
        embedding_new = self.model(image_align)

        random_label = self.random_n_digits(7)
        while random_label in labels:
            random_label = self.random_n_digits(7)

        random_label_array = np.array([random_label])

        # TODO: Fix loading and saving of the dictionary

        # names_dictionary[str(random_label)] = label
        # names_dictionary.update(self.dictionary)

        self.indexIDMap.add_with_ids(embedding_new, random_label_array)
        faiss_ppc.write_index(self.indexIDMap, "./classifier/vector.index")

        return "New person is added."
