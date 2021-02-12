from datetime import datetime
from time import perf_counter, time

import torch
from joblib import dump

from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1
from utils.vis_utils import extract_embeddings


def get_data():
    print("getting data...")
    timing = perf_counter()

    CLASSES_PER_BATCH = 30
    SAMPLES_PER_CLASS = 40
    BATCH_SIZE = CLASSES_PER_BATCH * SAMPLES_PER_CLASS

    # dataset = WebfaceDataset("../../data/CASIA-WebFace")
    dataset = WebfaceDataset("datasets/CASIA-WebFace")

    train_loader, _, _ = get_data_loaders(
        dataset,
        CLASSES_PER_BATCH,
        SAMPLES_PER_CLASS,
        train_proportion=0.01,
        val_proportion=0.89,
        test_proportion=0.1,
        batch_size=2000,
    )

    print("loading model...")

    if torch.cuda.is_available():
        checkpoint = torch.load(
            "checkpoints/generous-jazz-275_epoch_19",
            map_location=torch.device("cuda"),
        )
    else:
        checkpoint = torch.load(
            "checkpoints/generous-jazz-275_epoch_19",
            map_location=torch.device("cpu"),
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
        f"datasets/generous-jazz-275_epoch_19_{datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')}.joblib",
    )

    return embeddings, targets


if __name__ == "__main__":
    torch.manual_seed(42)

    embeddings, targets = get_data()
