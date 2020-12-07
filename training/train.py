import torch
from tqdm import tqdm

from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1

from .loss_function import triplet_loss


def train(model, train_loader, val_loader, loss_function, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1} of {num_epochs}")

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

        return (total_loss) / len(train_loader)


def evaluate(model, val_loader):
    pass


if __name__ == "__main__":
    if torch.cuda.is_available():
        model = InceptionResnetV1(torch.cuda.current_device())
    else:
        model = InceptionResnetV1()

    dataset = WebfaceDataset("datasets/CASIA-WebFace")
    train_loader, val_loader, _ = get_data_loaders(
        dataset,
        batch_size=1000,
        train_proportion=0.8,
        val_proportion=0.1,
        test_proportion=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, val_loader, triplet_loss, optimizer, num_epochs=100)
