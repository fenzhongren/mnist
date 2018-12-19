from mnist import training_image_datasets
from mnist import training_label_datasets
from mnist import testing_image_datasets
from mnist import testing_label_datasets

import numpy as np

num_classes = 10

np.seterr(all='raise')
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

# hypo is a k * num_examples matrix
def calculate_hypo(images, theta):
    #print(theta.T.shape, images.T.shape)
    z = np.dot(theta.T, images.T)
    max_per_examples = np.amax(z, axis=0, keepdims=True)
    z = z - max_per_examples;
    
    try:
        exp_scores = np.exp(z)
    except FloatingPointError:
        print(z)
        raise
    
    exp_sum_per_examples = np.sum(exp_scores, axis=0, keepdims=True)
    hypo = exp_scores/exp_sum_per_examples
    return hypo

def calculate_loss(images, labels, hypo):
    num_examples = len(images)
    right_per_example = hypo[labels, range(num_examples)]
    log_score = np.log(right_per_example)
    loss = -np.sum(log_score)
    return loss

def calculate_gradient(images, labels, hypo):
    hypoT = -hypo.T
    num_examples = len(images)
    
    y = labels.T
    temp = np.arange(num_classes).reshape(1, num_classes)
    adjust = (y == temp) * 1.0
    
    adjust_hypoT = adjust + hypoT
    dtheta = -np.dot(images.T, adjust_hypoT)
    return dtheta


def build_model(images, labels, num_pass=1000, step=0.01, print_loss=False):
    num_features = len(images[0])
    np.random.seed(0)
    theta = np.random.randn(num_features, num_classes)

    for i in range(num_pass):
        hypo = calculate_hypo(images, theta)
        print(hypo[:, 0])
        print(hypo[:, 1])
        loss = calculate_loss(images, labels, hypo)
        dtheta = calculate_gradient(images, labels, hypo)

        theta -= step*dtheta
        if print_loss == True:
            print(str(i) + " times loss: " + str(loss))

    return theta

theta = build_model(training_images, training_labels, step=0.1, print_loss=True)
