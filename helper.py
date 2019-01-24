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
        
















     
