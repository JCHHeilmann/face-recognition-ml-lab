from datetime import datetime
from time import perf_counter, time

import torch
from joblib import dump

from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1
from utils.vis_utils import extract_embeddings

# from pai4sk.svm import LinearSVC
# from sklearn.metrics import accuracy_score


# from sklearn.neighbors import RadiusNeighborsClassifier


def get_data():
    print("getting data...")
    timing = perf_counter()

    CLASSES_PER_BATCH = 30
    SAMPLES_PER_CLASS = 40

    dataset = WebfaceDataset("../../data/Aligned_CASIA_WebFace_JPG")
    # dataset = WebfaceDataset("datasets/CASIA-WebFace")

    train_loader, _, _ = get_data_loaders(
        dataset,
        CLASSES_PER_BATCH,
        SAMPLES_PER_CLASS,
        train_proportion=0.8,
        val_proportion=0.1,
        test_proportion=0.1,
        batch_size=2000,
    )

    print("loading model...")

    if torch.cuda.is_available():
        checkpoint = torch.load(
            "checkpoints/stilted-vortex-227_epoch_19", map_location=torch.device("cuda")
        )
    else:
        checkpoint = torch.load(
            "checkpoints/stilted-vortex-227_epoch_19", map_location=torch.device("cpu")
        )
    model = InceptionResnetV1()
    model.load_state_dict(checkpoint["model_state_dict"])

    if torch.cuda.is_available():
        model = model.cuda()

    print("calculating embeddings...")

    embeddings, targets = extract_embeddings(train_loader, model)

    print(f"took {perf_counter() - timing} seconds")

    dump(
        (embeddings, targets),
        f"../../data//embeddings_stilted-vortex-227_epoch_19_{datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')}.joblib",
        # f"../../data/embeddings_deft-snowball-123_epoch_19_{datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')}.joblib",
        # f"datasets/embeddings_charmed-cosmos-135_epoch_19_{datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')}.joblib",
    )

    return embeddings, targets


# def train_classifier(embeddings, targets):
#     print("training classifier...")
#     timing = perf_counter()

#     # classifier = RadiusNeighborsClassifier(radius=1, outlier_label=-1, n_jobs=-1)
#     classifier = LinearSVC(verbose=True)
#     print("initialized model")
#     classifier.fit(embeddings, targets)
#     print(f"took {perf_counter() - timing} seconds")

#     return classifier


def save_classifier(classifier):
    print("saving...")
    dump(
        classifier,
        f"face_recognition_classifier_{datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')}.joblib",
    )
    print("done")


# def evaluate_classifier(classifier, embeddings, targets):
#     print("testing...")
#     timing = perf_counter()

#     predictions = classifier.predict(embeddings)
#     accuracy = accuracy_score(targets, predictions)
#     print(f"train accuracy {accuracy}")

#     print(f"took {perf_counter() - timing} seconds")


if __name__ == "__main__":
    torch.manual_seed(42)

    embeddings, targets = get_data()
    # classifier = train_classifier(embeddings, targets)
    # save_classifier(classifier)
    # evaluate_classifier(classifier, embeddings, targets)
