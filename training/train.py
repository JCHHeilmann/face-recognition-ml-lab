from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import wandb
from classifier.faiss_classifier import FaissClassifier
from classifier.faiss_create import create_index
from data.data_loaders import get_data_loaders
from data.web_face_dataset import WebfaceDataset
from models.inception_resnet_v1 import InceptionResnetV1
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils.vis_utils import plot_embeddings

from training import triplet_generator
from training.loss_function import OnlineTripletLoss


def train(model, train_loader, val_loader, loss_function, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        print(f"epoch {epoch + 1} of {epochs}")

        # if not ((epoch + 1) % 4):
        #     margin *= 0.2

        # loss_function = OnlineTripletLoss(margin, triplet_generator.get_semihard)

        embeddings, targets = train_epoch(model, train_loader, loss_function, optimizer)

        eval_timing = perf_counter()
        total_accuracy, total_f1, total_accuracy_unknown, total_f1_unknown = evaluate(
            model, val_loader, loss_function
        )
        eval_timing = perf_counter() - eval_timing

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
                "eval_timing": eval_timing,
                "accuracy": total_accuracy,
                "accuracy_unknown": total_accuracy_unknown,
                "f1": total_f1,
                "f1_unknown": total_f1_unknown,
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


def evaluate(model, embeddings, targets, val_loader):

    index = create_index(embeddings.data.cpu().numpy(), targets.numpy())

    classifier = FaissClassifier(index)

    accuracy = 0
    accuracy_unknown = 0
    f1 = []
    f1_unknown = []
    predicted_labels = []
    val_labels = []

    with torch.no_grad():
        model.eval()

        for batch_idx, (data, target) in tqdm(
            enumerate(val_loader), total=len(val_loader), desc="processing batch: "
        ):
            val_labels.append(target.numpy())
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            outputs = model(data)
            predicted = classifier.classify(outputs.data.cpu().numpy())
            predicted_labels.append(classifier.classify(outputs.data.cpu().numpy()))

            accuracy = accuracy_score(target.numpy(), predicted)
            f1 = f1_score(target.numpy(), predicted, average=None)

            accuracy += accuracy
            f1 += f1

            # in val_labels replace all labels not in input targets with "Unknown"
            indx = np.where(np.in1d(predicted, targets.numpy()))[0]
            if len(indx) > 0:
                predicted[~np.array(indx)] = -1

            accuracy_unknown = accuracy_score(target.numpy(), predicted)
            f1_unknown = f1_score(target.numpy(), predicted, average=None)

            accuracy_unknown += accuracy_unknown
            f1_unknown += f1_unknown

        total_accuracy = accuracy / (batch_idx + 1)
        total_accuracy_unknown = accuracy_unknown / (batch_idx + 1)
        total_f1 = np.array(f1) / (batch_idx + 1)
        total_f1_unknown = np.array(f1_unknown) / (batch_idx + 1)

    return total_accuracy, total_f1, total_accuracy_unknown, total_f1_unknown


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

    model = InceptionResnetV1(
        DROPOUT_PROB, SCALE_INCEPTION_A, SCALE_INCEPTION_B, SCALE_INCEPTION_C
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)

    triplet_gen = triplet_generator.get_semihard

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
            "optimizer": str(optimizer),
            "scheduler": str(scheduler),
            "triplet_generation": str(triplet_gen),
            "margin": MARGIN,
        },
    )

    if torch.cuda.is_available():
        model = model.cuda()

    wandb.watch(model)

    # dataset = WebfaceDataset("../../data/Aligned_CASIA_WebFace")
    dataset = WebfaceDataset("datasets/CASIA-WebFace")

    train_loader, val_loader, _ = get_data_loaders(
        dataset,
        CLASSES_PER_BATCH,
        SAMPLES_PER_CLASS,
        train_proportion=0.8,
        val_proportion=0.1,
        test_proportion=0.1,
    )

    triplet_loss = OnlineTripletLoss(MARGIN, triplet_gen)
    train(
        model,
        train_loader,
        val_loader,
        triplet_loss,
        optimizer,
        scheduler,
        epochs=EPOCHS,
    )
