from mnist import training_image_datasets
from mnist import training_label_datasets
from mnist import testing_image_datasets
from mnist import testing_label_datasets

import numpy as np

def get_data_from_datasets(image_datasets, label_datasets):
    raw_images = image_datasets['images']
    images = (raw_images > 100) * 0.01
    num_examples = label_datasets['num_items']
    labels = label_datasets['labels'].reshape(1, num_examples)
    return (images, labels)

training_images, training_labels = get_data_from_datasets(
    training_image_datasets, training_label_datasets)

testing_images, testing_labels = get_data_from_datasets(
    testing_image_datasets, testing_label_datasets)


class NeuralNetworks:

    activations = ('sigmoid ', 'tanh', 'rectified linear')
    layers_error = 'Should have at least 3 layers!'
    unit_list_error = 'unist_list should match layers'
    check_features_error = 'Features should be the same as input layer units'
    activation_error = 'Only support activations sigmoid, '\
        'tanh and rectified linear'

    def check_parameters(self, layers, unit_list, act):
        assert layers >= 3, layers_error
        assert layers == len(unit_list), unit_list_error
        assert act in activations, activation_error
    
    def __init__(self, layers, unit_list, act):
        check_parameters(layers, unit_list, act)
        self.num_layers = layers
        self.unit_list = unit_list
        self.activation = act

    def check_features(self, num_features): 
        assert num_features == self.unit_list[0], check_features_error

    '''
    X is num_examples * num_features matrix
    Y is 1 * num_examples matrix
    '''
    def fit(self, X, Y):
        num_examples = len(X)
        num_features = len(X[0])

        check_features(num_features)
        
