import glob
import os
from os.path import join

import dlib
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset


class CasiaDataset(Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.read_file_paths()
        self.encode_classes()

    def read_file_paths(self):
        self.image_filenames = glob.glob(
            join(self.dataset_folder, "**/*.jpg"), recursive=True
        )

    def encode_classes(self):
        self.class_to_idx = dict()
        for filename in self.image_filenames:
            split_path = filename.split(os.sep)
            label = split_path[-2]
            self.class_to_idx[label] = self.class_to_idx.get(
                label, len(self.class_to_idx)
            )

    def shape_to_normal(self, shape):
        shape_normal = []
        for i in range(0, 5):
            shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
        return shape_normal

    def get_eyes_nose_dlib(self, shape):
        nose = shape[4][1]
        left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
        left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
        right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
        right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
        return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

    def feature_distance(self, a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def cosine_formula(self, length_line1, length_line2, length_line3):
        cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (
            2 * length_line2 * length_line1
        )
        return cos_a

    def rotate_point(self, origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def is_between(self, point1, point2, point3, extra_point):
        c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (
            point2[1] - point1[1]
        ) * (extra_point[0] - point1[0])
        c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (
            point3[1] - point2[1]
        ) * (extra_point[0] - point2[0])
        c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (
            point1[1] - point3[1]
        ) * (extra_point[0] - point3[0])
        if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
            return True
        else:
            return False

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx])
        image = image.resize((256, 256))
        split_path = self.image_filenames[idx].split(os.sep)
        label = split_path[-2]
        toTensor = torchvision.transforms.ToTensor()

        image = Image.open(self.image_filenames[idx])
        image = image.resize((256, 256))
        split_path = self.image_filenames[idx].split(os.sep)
        label = split_path[-2]

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

        toTensor = torchvision.transforms.ToTensor()

        image = np.array(image)

        rects = detector(image, 0)
        if len(rects) > 0:
            for rect in rects:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                shape = predictor(image, rect)

        shape = self.shape_to_normal(shape)
        nose, left_eye, right_eye = self.get_eyes_nose_dlib(shape)

        center_of_forehead = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        center_pred = (int((x + w) / 2), int((y + y) / 2))

        length_line1 = self.feature_distance(center_of_forehead, nose)
        length_line2 = self.feature_distance(center_pred, nose)
        length_line3 = self.feature_distance(center_pred, center_of_forehead)

        cos_a = self.cosine_formula(length_line1, length_line2, length_line3)
        angle = np.arccos(cos_a)

        rotated_point = self.rotate_point(nose, center_of_forehead, angle)
        rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
        if self.is_between(nose, center_of_forehead, center_pred, rotated_point):
            angle = np.degrees(-angle)
        else:
            angle = np.degrees(angle)

        image = Image.fromarray(image)
        image = np.array(image.rotate(angle))

        image = toTensor(image)

        return image, self.class_to_idx[label]

    def split(self):
        n_samples = len(self)
        shuffled_indices = np.random.permutation(n_samples)

        valandtest_ratio = 0.2
        testandvalidationset_inds = shuffled_indices[
            : int(n_samples * valandtest_ratio)
        ]
        trainingset_inds = shuffled_indices[int(n_samples * valandtest_ratio) :]

        test_ratio = 0.5
        n_samples_testandval = len(testandvalidationset_inds)
        validationset_inds = testandvalidationset_inds[
            : int(n_samples_testandval * test_ratio)
        ]
        testset_inds = testandvalidationset_inds[
            int(n_samples_testandval * test_ratio) :
        ]

        train_dataset = torch.utils.data.Subset(self, indices=trainingset_inds)
        val_dataset = torch.utils.data.Subset(self, indices=validationset_inds)
        test_dataset = torch.utils.data.Subset(self, indices=testset_inds)

        return train_dataset, val_dataset, test_dataset
