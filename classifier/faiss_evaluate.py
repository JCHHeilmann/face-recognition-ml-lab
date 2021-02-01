import joblib
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from classifier.faiss_classifier import FaissClassifier
from data.data_loaders import get_data_loaders
from data.face_alignment_mtcnn import FaceAlignmentMTCNN
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1


def evaluate(model, val_indices, train_labels):
    to_pil = torchvision.transforms.ToPILImage()
    preprocessor = FaceAlignmentMTCNN()

    # dataset = WebfaceDataset("datasets/CASIA-WebFace")

    classifier = FaissClassifier(
        index="datasets/vector_pre_trained_2021-02-01_14-42-55.index"
    )
    classifier.threshold = 100.0

    true = []
    pred = []

    if torch.cuda.is_available():
        model.cuda()

    with torch.no_grad():
        model.eval()

        for _, index in tqdm(
            enumerate(val_indices), total=len(val_indices), desc="evaluating batch: "
        ):
            image, target = dataset.get_file(index)

            if target not in list(train_labels):
                continue

            aligned_data = preprocessor.make_align(image)
            if aligned_data == None:
                continue

            if torch.cuda.is_available():
                aligned_data = aligned_data.cuda()

            outputs = model(aligned_data.unsqueeze(0))
            predicted = classifier.classify(outputs.cpu().numpy())

            print("target:", target, "Predicted:", np.array([predicted]))

            # decided to put this one in list so that F1 score can be calculated
            true.append(int(target))
            pred.append(predicted)

    total_accuracy = accuracy_score(np.array(true), np.array(pred))
    total_f1 = f1_score(np.array(true), np.array(pred), average="weighted")

    print(total_accuracy)
    return total_accuracy, total_f1


if __name__ == "__main__":
    torch.manual_seed(42)

    _, train_labels = joblib.load(
        "../../data/generous-jazz-275_epoch_19_2021-02-01_14-41-10.joblib"
    )

    checkpoint = torch.load(
        # "checkpoints/charmed-cosmos-135_epoch_19",
        # "checkpoints/major-cloud-212_epoch_19",
        "checkpoints/generous-jazz-275_epoch_19",
        map_location=torch.device("cpu"),
    )
    model = InceptionResnetV1()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # dataset = WebfaceDataset("datasets/CASIA-WebFace")
    dataset = WebfaceDataset("../../data/CASIA-WebFace")
    CLASSES_PER_BATCH = 35
    SAMPLES_PER_CLASS = 40
    BATCH_SIZE = CLASSES_PER_BATCH * SAMPLES_PER_CLASS
    print("Loading dataset...")
    train_loader, val_loader, _ = get_data_loaders(
        dataset,
        CLASSES_PER_BATCH,
        SAMPLES_PER_CLASS,
        train_proportion=0.8,
        val_proportion=0.1,
        test_proportion=0.1,
        batch_size=2000,
    )
    print("Dataset loaded")

    val_indices = val_loader.dataset.indices

    total_accuracy, total_f1 = evaluate(model, val_indices, train_labels)
    print("Accuracy:", total_accuracy)
    print("F1 score:", total_f1)
