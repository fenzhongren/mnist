from mnist import training_image_datasets
from mnist import training_label_datasets
from mnist import testing_image_datasets
from mnist import testing_label_datasets

import numpy as np
import csv

num_classes = 10

np.seterr(all='raise')
#np.seterr(divide='raise')

def save_matrix(ma):
    with open('save.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        size = len(ma)
        for i in range(size):
            writer.writerow(ma[i])

#images is a num_examples*num_features matrix
#labels is a 1*num_examples matrix
def get_data_from_datasets(image_datasets, label_datasets):
    images = image_datasets['images']
    softmax_images = (images > 100) * 0.01
    num_examples = label_datasets['num_items']
    softmax_labels = label_datasets['labels']
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
    z = z - max_per_examples*0.75;
    
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

#theta is num_features*num_classes matrix
def test_gradient(images, labels):
    num_features = len(images[0])
    np.random.seed(0)
    theta = np.random.randn(num_features, num_classes)
    
    gtheta = np.zeros(theta.shape)
    for n in range(num_features):
        for k in range(num_classes):
            epsilon = np.zeros(theta.shape)
            epsilon[n][k] = 0.0001
            theta_plus = theta + epsilon
            theta_minus = theta - epsilon
            hypo_plus = calculate_hypo(images, theta_plus)
            hypo_minus = calculate_hypo(images, theta_minus)
            loss_plus = calculate_loss(images, labels, hypo_plus)
            loss_minus = calculate_loss(images, labels, hypo_minus)
            gtheta[n][k] = (loss_plus - loss_minus)/0.0002

    hypo = calculate_hypo(images, theta)
    dtheta = calculate_gradient(images, labels, hypo)

    gtheta = np.around(gtheta, 4)
    dtheta = np.around(dtheta, 4)
    save_matrix((dtheta == gtheta))
    

def build_model(images, labels, num_pass=10000, step=0.01, print_loss=False):
    num_features = len(images[0])
    np.random.seed(0)
    theta = np.random.randn(num_features, num_classes)

    for i in range(num_pass):
        hypo = calculate_hypo(images, theta)
        #print(hypo[:, 0])
        #print(hypo[:, 1])
        loss = calculate_loss(images, labels, hypo)
        dtheta = calculate_gradient(images, labels, hypo)

        theta -= step*dtheta
        if print_loss == True:
            print(str(i) + " times loss: " + str(loss))

    return theta

#theta = build_model(training_images, training_labels, step=0.21, print_loss=True)
#test_gradient(training_images, training_labels)


class SoftMax:

    check_parameters_error = 'Input images or labels error'
    did_not_fit_error = "didn't fit your model"
    testing_data_not_match_error = 'Testing dataset error'
    
    
    def __init__(self):
        self.theta = None
        self.label_classes = None

    def check_training_parameters(self, X, Y):
        num_examples_X, num_features = X.shape
        num_examples_Y, = Y.shape
        assert num_examples_X == num_examples_Y, check_parameters_error

    def check_testing_parameters(self, X):
        assert (self.theta != None) and (self.label_classes != None), \
            did_not_fit_error
        num_testing_examples, num_testing_features = X.shape
        num_features, num_classes = self.theta.shape
        assert num_testing_features == num_features, \
            testing_data_not_match_error

    def relabel_by_index(self, Y):
        num_examples, = Y.shape
        self.label_classes = np.array(list(set(Y)))
        
        indices = np.zeros(num_examples)
        for i in range(len):
            index, = np.nonzero(self.label_classes==Y[i])
            indices[i] = index
        indices.shape = (1, num_examples)
        return indices

    def fit(self, X, Y, num_pass=10000, step=0.02, print_loss=True):
    '''
    Parameters
    ----------
    X : num_examples * num_features matrix
    Y : (num_examples, ) matrix
    '''
        self.check_training_parameters(X, Y)
        indices = self.relabel_by_index(Y)
        self.theta = build_model(X, indices, num_pass, step, print_loss)

    def predict(self, X):
    '''
    Parameters
    ----------
    X : num_examples * num_features matrix
    
    Return : (num_examples, ) matrix
    '''
        self.check_testing_parameters(X)
        hypo = calculate_hypo(X, self.theta)
        hypo.shape = (len(hypo[0]), )
        return hypo

if __name__ == "__main__":
    image_test = SoftMax()
    image_test.fit(training_images, training_labels)
    hypo = image_test(testing_images)
    results = (hypo == testing_labels)
    right_count = np.count_nonzero(results)
    percent = (right_count / len(hypo)) * 100
    print("accuracy: %" + str(percent))
