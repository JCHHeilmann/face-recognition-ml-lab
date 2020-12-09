import torch
import wandb
from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1
from tqdm import tqdm

from .loss_function import triplet_loss


def train(model, train_loader, val_loader, loss_function, optimizer, epochs):
    for epoch in range(epochs):
        print(f"epoch {epoch + 1} of {epochs}")

        train_epoch(model, train_loader, loss_function, optimizer)

        evaluate(model, val_loader)


def train_epoch(model, train_loader, loss_function, optimizer):
    total_loss = 0

    for i, (anchor, positive, negative) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        if torch.cuda.is_available():
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

        optimizer.zero_grad()

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)
        total_loss += loss.item()
        loss.backward()

        optimizer.step()

        if i % (len(train_loader) // 10) == 0:  # 10 logs per batch
            wandb.log({"training_loss": total_loss / (i + 1)})

        return (total_loss) / len(train_loader)


def evaluate(model, val_loader):
    pass


if __name__ == "__main__":
    EPOCHS = 100
    BATCH_SIZE = 1000
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.6
    SCALE_INCEPTION_A = 0.17
    SCALE_INCEPTION_B = 0.10
    SCALE_INCEPTION_C = 0.20

    wandb.init(
        project="face-recognition",
        entity="application-challenges-ml-lab",
        mode="enabled",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout_prob": DROPOUT_PROB,
            "scale_inception_a": SCALE_INCEPTION_A,
            "scale_inception_b": SCALE_INCEPTION_B,
            "scale_inception_c": SCALE_INCEPTION_C,
        },
    )

    model = InceptionResnetV1(
        DROPOUT_PROB, SCALE_INCEPTION_A, SCALE_INCEPTION_B, SCALE_INCEPTION_C
    )
    if torch.cuda.is_available():
        model.cuda()

    wandb.watch(model)

    dataset = WebfaceDataset("datasets/CASIA-WebFace")
    train_loader, val_loader, _ = get_data_loaders(
        dataset,
        batch_size=BATCH_SIZE,
        train_proportion=0.8,
        val_proportion=0.1,
        test_proportion=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, val_loader, triplet_loss, optimizer, epochs=EPOCHS)

