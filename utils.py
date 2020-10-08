"""
Provides miscellaneous functionality like computation of metrics
and averages, as well as data augmentation.
"""

import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image
from statistics import mean


class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Recorder object stores metrics and updates itself after each epoch
class Recorder(object):
    def __init__(self, names):
        self.names = names
        self.record = {}
        for name in self.names:
            self.record[name] = []

    def update(self, vals):
        for name, val in zip(self.names, vals):
            self.record[name].append(val)


# This function is only used when calling PixelNet
def generate_rand_ind(labels, n_class, n_samples):
    n_samples_avg = int(n_samples/n_class)
    rand_ind = []
    for i in range(n_class):
        positions = np.where(labels.view(1, -1) == i)[1]
        if positions.size == 0:
            continue
        else:
            rand_ind.append(np.random.choice(positions, n_samples_avg))
    rand_ind = np.random.permutation(np.hstack(rand_ind))
    return rand_ind


# Computes the accuracy of training outputs by summing number of pixels
# in an output equal to its label, and dividing by total number of pixels.
def accuracy(predictions, labels):
    correct = predictions.eq(labels.cpu()).sum().item()
    acc = correct/np.prod(labels.shape)
    return acc


# Takes directory of images and outputs mean and standard deviation of their tensors
# This code runs in the utils.py script to supply the mean and standard deviation of
# the images image set for normalization. Normalization is a necessary step in NN
# training as it allows errors in segmentation to be weighted equally.
def average(root_dir):
    # Defines path to training images. Files must be structured in this way.
    image_dir = os.path.join(root_dir, 'train/images/')

    # Initialize empty lists which will contain average and std. dev. of each image tensor.
    avg_list = []
    std_list = []

    # Loop through all images in image directory defined above.
    for pic in os.listdir(image_dir):
        img = Image.open(image_dir + pic)
        tensor = TF.to_tensor(img)
        avg_list.append(torch.mean(tensor).tolist())
        std_list.append(torch.std(tensor).tolist())

    # Returns list with three equal entries for each channel of the RGB image.
    avg = [round(mean(avg_list), 3)] * 3
    std = [round(mean(std_list), 3)] * 3
    return avg, std


# Performs random transformations to both an image and its label. Includes
# random rotations, vertical and horizontal flips. Normalizes image with
# respect to mean and standard deviation computed by average function.
def get_transform(config, is_train):
    mean, std, is_aug = average(config['root'])[0], average(config['root'])[1], config['aug']

    # Augmentation only occurs during training and can be toggled in configuration
    if is_train and is_aug:
        transform_label = T.Compose([
            T.RandomRotation(45),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor()
        ])
    else:
        transform_label = T.Compose([
            T.ToTensor()
        ])

    transform_img = T.Compose([
        transform_label,
        T.Normalize(mean=mean, std=std)
    ])
    return transform_img, transform_label


# Computes metrics relevant to evaluation of image segmentation, including
# precision, recall, and accuracy, as well as class and mean IoU.
def metrics(conf_mat, verbose=True):
    c = conf_mat.shape[0]

    # Ignore dividing by zero error
    np.seterr(divide='ignore', invalid='ignore')

    # Divide diagonal entries of confusion matrix by sum of its columns and
    # rows to respectively obtain precision and recall.
    precision = np.nan_to_num(conf_mat.diagonal()/conf_mat.sum(0))
    recall = conf_mat.diagonal()/conf_mat.sum(1)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Initialize empty array for IoU computation
    IoUs = np.zeros(c)
    union_sum = 0

    # Loop through rows of confusion matrix; divide each diagonal entry by the
    # sum of its row and column (while avoiding double-counting that entry).
    for i in range(c):
        union = conf_mat[i, :].sum()+conf_mat[:, i].sum()-conf_mat[i, i]
        union_sum += union
        IoUs[i] = conf_mat[i, i]/union

    # Accuracy computed by dividing sum of confusion matrix diagonal with
    # the sum of the confusion matrix
    acc = conf_mat.diagonal().sum()/conf_mat.sum()
    IoU = IoUs.mean()

    # IoU of second class corresponds to that of Somas, which we record.
    class_iou = IoUs[1]
    if verbose:
        print('precision:', np.round(precision, 5), precision.mean())
        print('recall:', np.round(recall, 5), recall.mean())
        print('IoUs:', np.round(IoUs, 5), IoUs.mean())
        print('BF1 Score:', np.round(f1_score, 5), f1_score.mean())
    return acc, IoU, precision, recall, class_iou
