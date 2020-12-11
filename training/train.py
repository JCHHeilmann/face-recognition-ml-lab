import torch
import wandb
from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1
from tqdm import tqdm
from training import loss_function
from training import triplet_generator

def train(model, train_loader, val_loader, loss_function, optimizer, epochs):
    for epoch in range(epochs):
        print(f"epoch {epoch + 1} of {epochs}")

        train_epoch(model, train_loader, loss_function, optimizer)

        evaluate(model, val_loader)


def train_epoch(model, train_loader, loss_function, optimizer):
    total_loss = 0
    model.train()
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)

        target = target if len(target) > 0 else None

        if not type(data) in (tuple, list):
            data = (data,)

        optimizer.zero_grad()

        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs

        if target is not None:
            target = (target,)
            loss_inputs += target
        loss_outputs = loss_function(*loss_inputs)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    total_loss /= (batch_idx + 1)
    print("Total Loss Epoch", total_loss)
    return total_loss



def evaluate(model, val_loader):
    pass


if __name__ == "__main__":
    EPOCHS = 2
    BATCH_SIZE = 12
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.6
    SCALE_INCEPTION_A = 0.17
    SCALE_INCEPTION_B = 0.10
    SCALE_INCEPTION_C = 0.20
    MARGIN = 1

    wandb.init(
        project="face-recognition",
        entity="application-challenges-ml-lab",
        mode="disabled",
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

    dataset = WebfaceDataset("../datasets/webface")
    print(dataset)
    train_loader, val_loader, _ = get_data_loaders(
        dataset,
        batch_size=BATCH_SIZE,
        train_proportion=0.8,
        val_proportion=0.1,
        test_proportion=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    triplet_loss = loss_function.OnlineTripletLoss(MARGIN, triplet_generator.RandomNegativeTripletSelector(MARGIN))
    train(model, train_loader, val_loader, triplet_loss, optimizer, epochs=EPOCHS)

