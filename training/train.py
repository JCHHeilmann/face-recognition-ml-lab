from pathlib import Path
from time import perf_counter

import torch
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1
from training import triplet_generator
from training.loss_function import OnlineTripletLoss
from utils.vis_utils import plot_embeddings


def train(model, train_loader, val_loader, loss_function, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        print(f"epoch {epoch + 1} of {epochs}")

        # if not ((epoch + 1) % 4):
        #     margin *= 0.2

        # loss_function = OnlineTripletLoss(margin, triplet_generator.get_semihard)

        embeddings, targets = train_epoch(model, train_loader, loss_function, optimizer)

        evaluate(model, val_loader, loss_function)

        save_checkpoint(model, optimizer, epoch)

        scheduler.step()

        embedding_visualization_timing = perf_counter()
        fig = plot_embeddings(embeddings, targets)
        wandb.log(
            {
                "embedding_plot": fig,
                "embedding_visualization_timing": (
                    perf_counter() - embedding_visualization_timing
                ),
            }
        )


def train_epoch(model, train_loader, loss_function, optimizer):

    total_loss = 0
    model_forward_timing = 0
    loss_timing = 0
    loss_backward_timing = 0
    optimizer_step_timing = 0
    total_num_triplets = 0

    model.train()

    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc="processing batch: "
    ):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        timing = perf_counter()
        outputs = model(data)
        model_forward_timing += perf_counter() - timing

        timing = perf_counter()
        loss, num_triplets = loss_function(outputs, target)
        loss_timing += perf_counter() - timing

        if num_triplets == 0:
            continue

        total_loss += loss.item()
        total_num_triplets += num_triplets

        timing = perf_counter()
        loss.backward()
        loss_backward_timing += perf_counter() - timing

        timing = perf_counter()
        optimizer.step()
        optimizer_step_timing += perf_counter() - timing

        if batch_idx % 10 == 0:  # 10
            wandb.log(
                {
                    "training_loss": total_loss / (batch_idx + 1),
                    "model_forward_timing": model_forward_timing / (batch_idx + 1),
                    "loss_timing": loss_timing / (batch_idx + 1),
                    "loss_backward_timing": loss_backward_timing / (batch_idx + 1),
                    "optimizer_step_timing": optimizer_step_timing / (batch_idx + 1),
                    "average_num_triplets": total_num_triplets / (batch_idx + 1),
                }
            )

    return (
        outputs.detach(),
        target,
    )  # return final batch embeddings for visualization


def evaluate(model, val_loader, loss_function):

    val_loss = 0
    val_num_triplets = 0

    with torch.no_grad():
        model.eval()

        for batch_idx, (data, target) in tqdm(
            enumerate(val_loader), total=len(val_loader), desc="processing batch: "
        ):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            outputs = model(data)

            loss, num_triplets = loss_function(outputs, target)

            if num_triplets == 0:
                continue

            val_loss += loss.item()
            val_num_triplets += num_triplets

            if batch_idx % 10 == 0:  # 10
                wandb.log(
                    {
                        "validation_loss": val_loss / (batch_idx + 1),
                        "average_num_triplets": val_num_triplets / (batch_idx + 1),
                    }
                )

    return val_loss


def save_checkpoint(model, optimizer, epoch_num):
    checkpoint_folder = Path("checkpoints/")
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_folder / (wandb.run.name + f"_epoch_{epoch_num}")
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

    EPOCHS = 20
    LEARNING_RATE = 0.05
    DROPOUT_PROB = 0.6
    SCALE_INCEPTION_A = 0.17
    SCALE_INCEPTION_B = 0.10
    SCALE_INCEPTION_C = 0.20
    MARGIN = 0.2

    CLASSES_PER_BATCH = 30
    SAMPLES_PER_CLASS = 40
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
            "scheduler": "MultiStepLR",
            "triplet_generation": "hardest",
        },
    )

    model = InceptionResnetV1(
        DROPOUT_PROB, SCALE_INCEPTION_A, SCALE_INCEPTION_B, SCALE_INCEPTION_C
    )
    if torch.cuda.is_available():
        model = model.cuda()

    wandb.watch(model)

    dataset = WebfaceDataset("../../data/Aligned_CASIA_WebFace")
    # dataset = WebfaceDataset("datasets/CASIA-WebFace")

    train_loader, val_loader, _ = get_data_loaders(
        dataset,
        CLASSES_PER_BATCH,
        SAMPLES_PER_CLASS,
        train_proportion=0.8,
        val_proportion=0.1,
        test_proportion=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)
    triplet_loss = OnlineTripletLoss(MARGIN, triplet_generator.get_hardest)
    train(
        model,
        train_loader,
        val_loader,
        triplet_loss,
        optimizer,
        scheduler,
        epochs=EPOCHS,
    )
