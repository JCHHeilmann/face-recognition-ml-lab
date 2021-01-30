import os
from random import randint

import numpy as np
import torch
import torchvision

from data.label_names import LabelNames
from models.inception_resnet_v1 import InceptionResnetV1

if os.uname().machine == "ppc64le":
    import faiss_ppc as faiss
else:
    import faiss

    from data.face_alignment import FaceAlignment


class FaissClassifier:
    def __init__(self, index="datasets/vector_pre_trained.index", model=None) -> None:
        super().__init__()

        self.threshold = 0.001
        self.to_tensor = torchvision.transforms.ToTensor()
        self.indexIDMap = faiss.read_index(index)
        self.dictionary = LabelNames("data/data.p")
        if os.uname().machine != "ppc64le":
            self.preprocessor = FaceAlignment()

        if not model:
            self.checkpoint = torch.load(
                "checkpoints/major-cloud-212_epoch_19",
                map_location=torch.device("cpu"),
                # "checkpoints/stilted-vortex-227_epoch_19",
                # map_location=torch.device("cpu"),
                # "checkpoints/charmed-cosmos-135_epoch_19",
                # map_location=torch.device("cpu"),
                # "checkpoints/deft-snowball-123_epoch_19",
                # map_location=torch.device("cpu"),
            )
            self.model = InceptionResnetV1()
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        else:
            self.model = model

        self.model.eval()

    def img_tensor_to_encoding(self, image_tensor, model):
        image_tensor = image_tensor.unsqueeze(0)
        embedding = model(image_tensor)
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
        pred_labels = []
        k = 1
        distance, label = self.indexIDMap.search(embeddings.astype("float32"), k)
        if distance < self.threshold:
            return int(label)
        else:
            return 0

    def classify_with_surroundings(self, image):
        image_aligned = self.preprocessor.make_align(image)
        if not image_aligned:
            print("No face found")
            return ["Unknown"], None

        embedding = self.img_to_encoding(image_aligned, self.model)

        k = 100
        distances, labels, embeddings = self.indexIDMap.search_and_reconstruct(
            embedding.astype("float32"), k
        )

        distances = distances[0]
        labels = labels[0]
        embeddings = embeddings[0]

        if distances[0] < self.threshold:
            label_names = [self.dictionary.read_from_pickle(label) for label in labels]

            return label_names, embeddings
        else:
            return ["Unknown"], None

    def add_person(self, image, name: str):
        image_align = self.preprocessor.make_align(image)
        embedding_new = self.img_to_encoding(image_align, self.model)

        random_label = self.random_n_digits(8)
        random_label_array = np.array([random_label])

        self.indexIDMap.add_with_ids(embedding_new, random_label_array)
        self.dictionary.add_name(name, random_label)


if __name__ == "__main__":
    from glob import glob

    from PIL import Image
    from tqdm import tqdm

    # image_paths = glob("datasets/CASIA-WebFace/0000192/*.png")
    # images = [Image.open(path).convert("RGB") for path in image_paths]

    image_paths = glob("datasets/CASIA-WebFace_PNG/2838320/*.png")
    images = [Image.open(path).convert("RGB") for path in image_paths]

    classifier = FaissClassifier(index="datasets/vector_pre_trained.index")
    # images = [Image.open("datasets/Donald_Trump_0001.jpg").convert("RGB")]
    # im = Image.open("datasets/0000045_a/008.jpg").convert("RGB").resize((128,128))

    # classifier = FaissClassifier(index = "datasets/vector_pre_trained_2021-01-28_15:22:37.index")

    # new_image = Image.open("datasets/test/IMG-20140329-WA0010.jpg")

    # classifier.add_person(new_image, "victor")

    # labels, _ = classifier.classify_with_surroundings(new_image)
    # print(labels[0])

    # new_image_2 = Image.open("datasets/test/IMG-20140329-WA0013.jpg")
    # labels, _ = classifier.classify_with_surroundings(new_image_2)
    # print(labels[0])

    # labels = classifier.classify_with_surroundings(im)[0][0]
    # print("Labels:",labels)

    results = [classifier.classify_with_surroundings(img)[0][0] for img in tqdm(images)]

    correct = 0
    incorrect = 0
    unknown = 0
    for result in results:
        if result == "Morgan_Saylor":
            correct += 1
        elif result == "Unknown":
            unknown += 1
        else:
            incorrect += 1

    accuracy = correct / len(results)

    print(
        f"Results:\ncorrect: {correct},\nincorrect: {incorrect},\nunknown: {unknown} (includes images in which no face was found),\nAccuracy: {accuracy}"
    )

    values, counts = np.unique(results, return_counts=True)
    unique_counts = list(zip(values, counts))
