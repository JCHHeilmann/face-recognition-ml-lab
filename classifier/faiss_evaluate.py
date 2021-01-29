import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from classifier.faiss_classifier import FaissClassifier
from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1


def evaluate(model, val_loader, train_loader):

    classifier = FaissClassifier(index="datasets/vector_pre_trained.index")
    classifier.threshold = 0.02

    true = []
    pred = []

    if torch.cuda.is_available():
        model.cuda()

    with torch.no_grad():
        model.eval()

        for _, (data, target) in tqdm(
            enumerate(val_loader), total=len(val_loader), desc="evaluating batch: "
        ):

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            outputs = model(data)
            predicted = classifier.classify(outputs.data.cpu().numpy())

            target = target.cpu()
            print("target:", target.numpy(), "Predicted:", np.array([predicted]))

            # decided to put this one in list so that F1 score can be calculated
            true.append(int(target.numpy()))
            pred.append(predicted)

    total_accuracy = accuracy_score(np.array(true), np.array(pred))
    total_f1 = f1_score(np.array(true), np.array(pred), average="weighted")

    print(total_accuracy)
    return total_accuracy, total_f1


if __name__ == "__main__":

    torch.manual_seed(42)
    checkpoint = torch.load(
        # "checkpoints/charmed-cosmos-135_epoch_19",
        "checkpoints/major-cloud-212_epoch_19",
        map_location=torch.device("cpu"),
    )
    model = InceptionResnetV1()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # dataset = WebfaceDataset("datasets/Aligned_CASIA_WebFace")
    dataset = WebfaceDataset("../../data/Aligned_CASIA_WebFace")
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
    )
    print("Dataset loaded")

    total_accuracy, total_f1 = evaluate(model, val_loader, train_loader)
    print("Accuracy:", total_accuracy)
    print("F1 score:", total_f1)
