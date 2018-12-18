import mnist
from mnist import training_image_datasets
from mnist import training_label_datasets
from mnist import testing_image_datasets
from mnist import testing_label_datasets
import numpy as np
import matplotlib.pyplot as plt

#np.set_printoptions(threshold=100)

def get_logistic_datasets(images, labels):
    indices, = np.nonzero(labels<2)
    logistic_labels = labels[indices]
    logistic_labels =\
        logistic_labels.reshape(len(logistic_labels), 1)
    logistic_images = images[indices]
    return (logistic_images, logistic_labels)

training_images = training_image_datasets['images']
training_labels = training_label_datasets['labels']
logistic_training_images, logistic_training_labels = \
    get_logistic_datasets(training_images, training_labels)

testing_images = testing_image_datasets['images']
testing_labels = testing_label_datasets['labels']
logistic_testing_images, logistic_testing_labels = \
    get_logistic_datasets(testing_images, testing_labels)


def show_image(image, label):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(str(label))
    plt.show()
    
# show_image(range(8))


def convert_image(images, value=255):
    '''
    with np.nditer(images, op_flags=['readwrite']) as it:
        for x in it:
            if x>100:
                x[...] = value
            else:
                x[...] = 0
    '''
    intercept_term = np.ones((len(images), 1))
    images = np.r_['1, 2', images, intercept_term]
    return images

def calculate_hypo(images, theta):
    z = images.dot(theta)
    exp_score = np.exp(-z)
    hypo = 1/(1+exp_score)
    return hypo

def calculate_error_rate(images, labels, theta):
    hypo = calculate_hypo(images, theta)
    compare = (labels == hypo)
    right_count = np.count_nonzero(compare)
    error_rate = 1 - right_count/len(labels)
    return error_rate


def build_logistic_module(images, labels, num_passes=1000, step=0.01,\
                          print_error=False):
    num_features = len(images[0])
    np.random.seed(0)
    
    theta = np.random.randn(num_features, 1)

    for i in range(num_passes):
        hypo = calculate_hypo(images, theta)

        delta = hypo - labels
        dtheta = np.dot(images.T, delta)
        
        theta -= step*dtheta
        error_rate = calculate_error_rate(images, labels, theta)
        if print_error == True:
            print(str(i) + ' times: %' + str(error_rate*100) + ' error')

        if error_rate < 0.0008:
            break
        
    return theta

def predict(images, theta):
    hypo = calculate_hypo(images, theta)
    with np.nditer(hypo, op_flags=['readwrite']) as it:
        for x in it:
            if x>0.5:
                x[...] = 1
            else:
                x[...] = 0

    return hypo

    

logistic_training_images = convert_image(logistic_training_images, 1)
logistic_testing_images = convert_image(logistic_testing_images, 1)
theta = build_logistic_module(logistic_training_images, \
                              logistic_training_labels, print_error=True)

hypo = predict(logistic_testing_images, theta)
compare = (logistic_testing_labels == hypo)
right_count = np.count_nonzero(compare)
error_rate = 1 - right_count/len(logistic_testing_labels)
print('%' + str(error_rate*100) + ' error')

#show_image(range(100, 200, 10))
