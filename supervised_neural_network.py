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

def extract_label_classes_and_indices_from_labels(labels):
    '''
    Extract how many kinds of labels(label_classes) from provided labels. 
    Use the label_classes to index labels so that we can covert provided 
    labels to integer-form labels range from 0 to kinds-of-label - 1.
    kinds-of-label equals to len(label_classes)
    
    Reture
    ----------
    label_classes: (kinds-of-label, ) matrix
    indices = (1, num_examples) matrix
    '''
    num_examples, = labels.shape
    label_classes = np.array(list(set(labels)))

    indices = np.zeros(num_examples, dtype=int).reshape(1, num_examples)
    for i in range(num_examples):
        index, = np.nonzero(label_classes==labels[i])
        indices[0][i] = index
    return (label_classes, indices)
    
    

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
       
    def build_model(self, examples):

        outputs = []
        outputs.append(examples.T)
        activation = helper.create_activation(sefl.act)
        
        # calculate hidden layer, num_layers =4, i = 0, 1
        for i in range(self.num_layers-2):
            z_plus = np.dot(self.W[i], outputs[i]) + self.b[i]
            a_plus = activation.calculate(z_plus)
            outputs.append(a.plus)

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
        hypo = exp_scores/exp_sum_per_examples

        
    
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
            extract_label_classes_and_indices_from_labels(labels)
        check_fit_features(label_classes, num_features)
        
        initialize_model();
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
