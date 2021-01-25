from random import randint

# import faiss_ppc as faiss
import faiss
import numpy as np
import torch
import torchvision

from data.face_alignment import FaceAlignment
from data.label_names import LabelNames
from models.inception_resnet_v1 import InceptionResnetV1


class FaissClassifier:
    def __init__(self, index="datasets/vector.index") -> None:
        super().__init__()

        self.threshold = 0.1
        self.to_tensor = torchvision.transforms.ToTensor()
        # self.indexIDMap = faiss.read_index("datasets/vector.index")
        self.indexIDMap = faiss.read_index(index)
        self.dictionary = LabelNames("data/data.p")
        self.preprocessor = FaceAlignment()

        self.checkpoint = torch.load(
             "checkpoints/charmed-cosmos-135_epoch_19", map_location=torch.device("cpu"),
            # "checkpoints/deft-snowball-123_epoch_19",
            # map_location=torch.device("cpu"),
        )
        self.model = InceptionResnetV1()
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

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
        for indx, d in enumerate(distance):
            if d < self.threshold:
                pred_labels.append(label[indx])
            else:
                pred_labels.append(-1)
        
        return pred_labels

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

    image_paths = glob("datasets/test/0000107/*.jpg")[:12]
    images = [Image.open(path).convert("RGB") for path in image_paths]

    classifier = FaissClassifier()

    new_image = Image.open("datasets/test/IMG-20140329-WA0010.jpg")

    classifier.add_person(new_image, "victor")

    label = classifier.classify(new_image)
    print(label)

    labels, embeddings = classifier.classify_with_surroundings(images[0])

    results = [classifier.classify(img) for img in tqdm(images)]

    correct = 0
    incorrect = 0
    unknown = 0
    for result in results:
        if result == "Kim_Basinger":
            correct += 1
        elif result == "Unknown":
            unknown += 1
        else:
            incorrect += 1

    print(
        f"Results:\ncorrect: {correct},\nincorrect: {incorrect},\nunknown: {unknown} (includes images in which no face was found)\n"
    )

    values, counts = np.unique(results, return_counts=True)
    unique_counts = list(zip(values, counts))
