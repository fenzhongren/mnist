from mnist import training_image_datasets
from mnist import training_label_datasets
from mnist import testing_image_datasets
from mnist import testing_label_datasets
import helper

import numpy as np


def get_data_from_datasets(image_datasets, label_datasets):
    '''
    Reture
    ----------
    images: (num_examples, num_features) matrix
    labels: (num_examples, ) matrix
    '''
    raw_images = image_datasets['images']
    images = (raw_images > 100) * 0.01
    num_examples = label_datasets['num_items']
    labels = label_datasets['labels']
    return (images, labels)

training_images, training_labels = get_data_from_datasets(
    training_image_datasets, training_label_datasets)

testing_images, testing_labels = get_data_from_datasets(
    testing_image_datasets, testing_label_datasets)

class SupervisedNeuralNetwork:
    '''
    Create a supervised neural network describeed in:
  http://ufldl.stanford.edu/tutorial/supervised/ExerciseSupervisedNeuralNetwork/
  
    SupervisedNeuralNetwork(layout, units, act)
    Parameters
    ---------- 
    layoyt: A tuple denotes all layers and how many units each layer has.
    act: A string denotes ste activation we use in hidden layer. 
        Currently, the supported activations are "sigmoid", "tanh" and 
        "rectified linear". Note that we always ues softmax as activation 
        for output layer.
    '''

    activations = ('sigmoid ', 'tanh', 'rectified linear')
    common_error = "\nPlease see the class docstrings!"
    layers_error = 'Should have at least have 3 layers.'
    check__features_error = 'Features should be the same as input layer units.'\
        ' kinds of label should less than or equal to output units'
    activation_error = 'Only support activations sigmoid, '\
        'tanh and rectified linear.'
    label_classes_error = 'The kinds if label must be less than or equal to' \
        'output layer.'

    def check_init_parameters(self, layout, act):
        assert len(layout) >= 3, layers_error + common_error
        assert act in activations, activation_error + common_error
    
    def __init__(self, layout, act):
        check_init_parameters(layout, act)
        self.num_layers = len(layout)
        self.layout = layout
        self.act = act
        sefl.label_classes = None

    def check_fit_features(self, label_classes, num_features): 
        assert num_features == self.layout[0], check_features_error
        assert len(label_classes) <= self.units[self.num_layers - 1], \
            label_classes_error

    def initialize_model(self):
        self.W = []
        self.b = []
        np.random.seed(0)
        for i in range (self.num_layers - 1):
            weight = 0.01 * random.randn(self.layout[i+1], layout[i])
            bias = 0.01 * random.randn(self.layout[i+1])
            self.W.append(weight)
            self.b.append(bias)
    
    # assume we have a 4-layer network: 0, 1, 2, 3
    def build_model(self, examples, num_pass, step, lmd):
        num_examples, num_features = examples.shape
        # outputs contains activation results [0, 1, 2]
        outputs = []
        # input layer, layer:0
        outputs.append(examples.T)
        activation = helper.create_activation(sefl.act)
        
        # calculate hidden layer, num_layers =4, i = 0, 1
        for j in range(num_pass):
            for i in range(self.num_layers-2):
                z_plus = np.dot(self.W[i], outputs[i]) + self.b[i]
                a_plus = activation.calculate(z_plus)
                outputs.append(a_plus)

            # for output layer, we use softmax
            temp = self.num_layers-2
            z_plus = np.dot(self.W[temp], output[temp]) + self.b[temp]
            max_per_examples = np.amax(z_plus, axis=0, keepdims=True)
            z_plus = z_plus - max_per_examples*0.75;

            try:
                exp_scores = np.exp(z_plus)
            except FloatingPointError:
                print(z_plus)
                raise

            exp_sum_per_examples = np.sum(exp_scores, axis=0, keepdims=True)
            # hypo is a (output_units, num_examples) maxtrix
            hypo = exp_scores/exp_sum_per_examples

            # perform backpropagation for output layer: 3
            # deltas contains deltas [1, 2, 3]
            deltas = []
            delta = hypo
            delta[self.indices[0], range(num_examples)] -= 1
            delta = np.sum(delta, axis=1, keepdims=True)
            deltas.insert(0, delta)
            # perform backpropagation for hidden layer
            # num_layers =4, we will perform for layer2 and layer1
            layers = list(range(self.num_layers-2, 0 -1))   #[2, 1]
            for i in range(self.num_layers-2):
                delta = np.dot(self.W[i].T, deltas[0]) * \
                    calculate_derivative_by_activation(outputs[i])
                deltas.insert(0, delta)

            # calculate gradient [0, 1, 2]
            for i in range(self.num_layers - 1):
                dW = np.dot(deltas[i], outputs[i].T)
                db = deltas[i]
                self.W[i] -= step * (dW/num_examples + lmd*self.W[i])
                self.b[i] -= step * db / num_examples

    '''
    Parameters
    ----------
    examples : (num_examples, num_features) matrix
    labels : (num_examples, ) matrix
    '''
    def fit(self, examples, labels, num_pass=100000, step=0.02, \
        print_loss=True):
        num_examples, num_features = examples.shape
        label_classes, indices = \
            helper.extract_label_classes_and_indices_from_labels(labels)
        check_fit_features(label_classes, num_features)
        self.label_classes, self.indices = label_classes, indices
        
        initialize_model();
        build_model(examples, num_pass, step, lmd=0.01)
        