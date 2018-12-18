import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def get_images(image_file_name):
    with open(image_file_name, 'br') as f:
        buf = f.read(16)
        magic_number, num_images, num_rows, num_columns = \
            struct.unpack('>4l', buf)

        pixel_per_image = num_rows * num_columns
        total_pixel_of_images = num_images * pixel_per_image
        
        buf = f.read(total_pixel_of_images)
        unpacked_images = struct.unpack('>'+ str(total_pixel_of_images) + 'B', \
                                      buf)
        images = np.array(unpacked_images).reshape(num_images, pixel_per_image)
        image_datasets = {'num_images': num_images, 'num_rows': num_rows, \
                        'num_columns': num_columns, 'images': images}

        return image_datasets

def get_labels(label_file_name):
    with open(label_file_name, 'br') as f:
        buf = f.read(8)
        magic_number, num_items = struct.unpack('>2l', buf)
        
        buf = f.read(num_items)
        unpacked_labels = struct.unpack('>' + str(num_items) + 'B', buf)
        labels = np.array(unpacked_labels)

        label_datasets = {'num_items': num_items, 'labels': labels}
        return label_datasets

training_image_datasets = \
    get_images('train-images-idx3-ubyte/train-images.idx3-ubyte')
training_images = training_image_datasets['images']
training_num_images = training_image_datasets['num_images']
training_num_rows = training_image_datasets['num_rows']
training_num_columns = training_image_datasets['num_columns']

training_label_datasets = \
    get_labels('train-labels-idx1-ubyte/train-labels.idx1-ubyte')
training_labels = training_label_datasets['labels']
training_num_items = training_label_datasets['num_items']

testing_image_datasets = \
    get_images('t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
testing_images = testing_image_datasets['images']

testing_label_datasets = \
    get_labels('t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')
testing_labels = testing_label_datasets['labels']

def show_image(number, label=None):
    plt.imshow(training_images[number].reshape(28, 28), cmap='gray')
    if(label == None):
        plt.title(str(training_labels[number]))
    else:
        plt.title(str(label))
    plt.show()

if __name__ == "__main__":
    num_examples = len(training_images)
    nn_input_dim = training_num_rows * training_num_columns
    nn_output_dim = 10

    epsilon = 0.01
    reg_lambda = 0.01

    def build_module(nn_hdim, num_passes=20000, print_loss=False):
        np.random.seed(0)
    
        W1 = np.random.randn(nn_input_dim, nn_hdim)/np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim)
        b2 = np.zeros((1, nn_output_dim))

        for i in range(num_passes):
            print('i = ' + str(i))
            z1 = training_images.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            print(b2)
            exp_scores = np.exp(z2)
            a2 = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

            delta3 = a2
            delta3[range(num_examples), training_labels] -= 1
            dW2 = np.dot(a1.T, delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = (1 - a1**2) * np.dot(delta3, W2.T)
            dW1 = np.dot(training_images.T, delta2)
            db1 = np.sum(delta2, axis=0, keepdims=True)

            dW1 += reg_lambda * W1
            dW2 += reg_lambda * W2

            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2

            module = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        return module

    def predict(module, image):
        W1, b1, W2, b2 = module['W1'], module['b1'], module['W2'], module['b2']

        z1 = image.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        y = np.argmax(z2)

        return y


    module = build_module(int(nn_input_dim*1.5))
    y = predict(module, training_images[0])
    show_image(0, y)
