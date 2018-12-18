from mnist import training_image_datasets
from mnist import training_label_datasets
from mnist import testing_image_datasets
from mnist import testing_label_datasets

import numpy as np

#images is a num_examples*num_features matrix
#labels is a 1*num_examples matrix
def get_data_from_datasets(image_datasets, label_datasets):
    softmax_images = image_datasets['images']
    num_examples = label_datasets['num_items']
    softmax_labels = label_datasets['labels'].reshape(1, num_examples)
    return (softmax_images, softmax_labels)

training_images, training_labels = get_data_from_datasets(
    training_image_datasets, training_label_datasets)

testing_images, testing_labels = get_data_from_datasets(
    testing_image_datasets, testing_label_datasets)

def calculate_hypo(images, theta):
    z = images.dot(theta)
    max_per_example = np.amax(z, axis=1, keepdims=True)
    z = z - max_per_examples*0.75;
    exp_scores = np.exp(z)
    exp_sum_per_example = np.sum(exp_scores, axis=1, keepdims=True)
    hypo = exp_scores/exp_sum_per_examples
    return hypo

def build_model(images, labels, num_pass=1000, step=0.01, print_loss=False):
    num_features = len(images[0])
    num_label_classes = len(labels[0])
    np.random.seed(0)
    theta = np.random.randn(num_features, num_label_classes)

    for i in range(num_pass):
        hypo = calculate_hypo(images, theta)
        