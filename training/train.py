from pathlib import Path

import torch
import wandb
from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1
from tqdm import tqdm
from utils.vis_utils import extract_embeddings, plot_embeddings

from training import triplet_generator
from training.loss_function import OnlineTripletLoss


def train(model, train_loader, val_loader, loss_function, optimizer, epochs):
    for epoch in range(epochs):
        print(f"epoch {epoch + 1} of {epochs}")

        train_epoch(model, train_loader, loss_function, optimizer)

        evaluate(model, val_loader)

        save_checkpoint(model, epoch)

        train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
        plt = plot_embeddings(train_embeddings_otl, train_labels_otl)
        wandb.log({"embedding_plot": plt})

        # model_copy = copy.deepcopy(model).cpu()
        # executor = ThreadPoolExecutor()
        # executor.submit(plot_and_log_embeddings, train_loader, model)
        # executor.shutdown(wait=False)


def train_epoch(model, train_loader, loss_function, optimizer):
    total_loss = 0
    model.train()
    losses = []

    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="processing batch: "
    ):
        target = target if len(target) > 0 else None

        if not type(data) in (tuple, list):
            data = (data,)

        if torch.cuda.is_available():
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()

        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs

        if target is not None:
            target = (target,)
            loss_inputs += target
        loss_outputs = loss_function(*loss_inputs)

        if loss_outputs is None:
            continue

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:  # 10
            wandb.log({"training_loss": total_loss / (batch_idx + 1)})

    total_loss /= batch_idx + 1
    print("Total Loss Epoch", total_loss)
    return total_loss


def evaluate(model, val_loader):
    pass


def save_checkpoint(model, optimizer, epoch_num):
    checkpoint_folder = Path("checkpoints/")
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_folder / (wandb.run.name + f"epoch_{epoch_num}")
    torch.save(
        {
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_name": wandb.run.name,
        },
        checkpoint_file,
    )
    wandb.save(str(checkpoint_file))


if __name__ == "__main__":
    torch.manual_seed(42)

    EPOCHS = 10
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.6
    SCALE_INCEPTION_A = 0.17
    SCALE_INCEPTION_B = 0.10
    SCALE_INCEPTION_C = 0.20
    MARGIN = 1

    CLASSES_PER_BATCH = 10
    SAMPLES_PER_CLASS = 25
    BATCH_SIZE = CLASSES_PER_BATCH * SAMPLES_PER_CLASS

    wandb.init(
        project="face-recognition",
        entity="application-challenges-ml-lab",
        # mode="disabled",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "classes_per_batch": CLASSES_PER_BATCH,
            "samples_per_class": SAMPLES_PER_CLASS,
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
        model = model.cuda()

    wandb.watch(model)

    # dataset = WebfaceDataset("../../data/CASIA-WebFace")
    dataset = WebfaceDataset("datasets/CASIA-WebFace")

    train_loader, val_loader, _ = get_data_loaders(
        dataset,
        CLASSES_PER_BATCH,
        SAMPLES_PER_CLASS,
        train_proportion=0.01,
        val_proportion=0.89,
        test_proportion=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    triplet_loss = OnlineTripletLoss(MARGIN, triplet_generator.get_semihard)
    train(model, train_loader, val_loader, triplet_loss, optimizer, epochs=EPOCHS)
