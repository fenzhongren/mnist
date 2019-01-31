import numpy as np

class Activation:

    def calculate(self, X):
        pass    
    def calculate_derivative_by_activation(self, act):
        pass
        
class Sigmoid(Activation):
    def calculate(self, X):
        return 1 / (1 + np.exp(X))
       
    def calculate_derivative_by_activation(self, act):
        return act.dot(1 - act)
        
class Tanh(Activation):
    def calculate(self, X):
        return np.tanh(X)
       
    def calculate_derivative_by_activation(self, act):
        return (1 - np.power(act, 2))
        
class RectifiedLinear(Activation):
    def calculate(self, X):
        compare = (X >= 0)
        zeros = np.zeros(X.shape)
        return np.where(compare, X, zeros)
       
    def calculate_derivative_by_activation(self, act):
        zeros = np.zeros(X.shape)
        ones = np.ones(X.shape)
        return np.where(act, ones, zeros)

activation_kind_err = 'Activation just support sigmoid, tanh and rectified linear'
  
def create_activation(act):
    if act == 'sigmoid':
        return Sigmoid()
    else if act == 'tanh':
        return tanh()
    else if act == 'rectified linear':
        return RectifiedLinear()
    else:
        assert 0, activation_kind_err
        
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
    num_classes = len(label_classes)

    class_base_row = label_classes.reshape(1, num_classes)
    class_base_column = labels.reshape(num_examples, 1)
    X, Y = np.nonzero(class_base_row == class_base_column)
    indices = Y.reshape(1, num_examples)
    return (label_classes, indices)