from pathlib import Path

import torch
import wandb
from tqdm import tqdm

from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1
from training import triplet_generator
from training.loss_function import OnlineTripletLoss
from utils.vis_utils import extract_embeddings, plot_embeddings


def train(model, train_loader, val_loader, loss_function, optimizer, epochs):
    for epoch in range(epochs):
        print(f"epoch {epoch + 1} of {epochs}")

        train_epoch(model, train_loader, loss_function, optimizer)

        evaluate(model, val_loader)

        save_model(model, epoch)

        train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
        plt = plot_embeddings(train_embeddings_otl, train_labels_otl)
        wandb.log({"embedding_plot": plt})


def train_epoch(model, train_loader, loss_function, optimizer):
    total_loss = 0
    model.train()
    losses = []

    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
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

        if batch_idx % (len(train_loader) // 10) == 0:  # 10 logs per batch
            wandb.log({"training_loss": total_loss / (batch_idx + 1)})

    total_loss /= batch_idx + 1
    print("Total Loss Epoch", total_loss)
    return total_loss


def evaluate(model, val_loader):
    pass


def save_model(model, epoch_num):
    model_folder = Path("models/")
    model_folder.mkdir(parents=True, exist_ok=True)
    model_file = model_folder / wandb.run.name + f"epoch_{epoch_num}"
    torch.save(model.state_dict(), model_file)
    wandb.save(str(model_file))


if __name__ == "__main__":
    torch.manual_seed(42)

    EPOCHS = 2
    BATCH_SIZE = 200
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.6
    SCALE_INCEPTION_A = 0.17
    SCALE_INCEPTION_B = 0.10
    SCALE_INCEPTION_C = 0.20
    MARGIN = 1

    wandb.init(
        project="face-recognition",
        entity="application-challenges-ml-lab",
        # mode="disabled",
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
        train_proportion=0.01,
        val_proportion=0.89,
        test_proportion=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    triplet_loss = OnlineTripletLoss(MARGIN, triplet_generator.get_semihard)
    train(model, train_loader, val_loader, triplet_loss, optimizer, epochs=EPOCHS)
