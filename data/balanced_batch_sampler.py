import numpy as np
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, classes_per_batch, samples_per_class):
        self.labels = labels
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.dataset_size = len(self.labels)
        self.batch_size = self.classes_per_batch * self.samples_per_class

        self.labels_set = list(set(self.labels))
        self.label_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0

    def __iter__(self):
        self.count = 0

        while self.count + self.batch_size < self.dataset_size:
            print(len(self.labels_set), len(self.classes_per_batch))
            classes = np.random.choice(
                self.labels_set, self.classes_per_batch, replace=False
            )
            indices = []

            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.samples_per_class
                    ]
                )
                self.used_label_indices_count[class_] += self.samples_per_class

                if self.used_label_indices_count[class_] + self.samples_per_class > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += self.classes_per_batch * self.samples_per_class

    def __len__(self):
        return self.dataset_size // self.batch_size
