import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train.astype('float32')/255, x_test.astype('float32')/255
    x_train, x_test = x_train.reshape(60000,784).T, x_test.reshape(10000,784).T
    y_train, y_test = one_hot(y_train), one_hot(y_test)
    return (x_train, y_train), (x_test, y_test)

def init_params():
    w1 = np.random.randn(128,784) +0.5
    b1 = np.random.randn(128,) + 0.5
    w2 = np.random.randn(10,128) + 0.5
    b2 = np.random.randn(10,) + 0.5
    return w1,b1,w2,b2

def ReLU(x):
    return np.maximum(0,x)

def ReLU_deriv(x):
    return x>0

def softmax(x):
    exp = np.exp(x-np.max(x))
    return exp/np.sum(exp)

def one_hot(y):
    nrows = 10
    ncols = len(y)
    one_hot_y = np.zeros((nrows,ncols))
    one_hot_y[y,np.arange(ncols)] = 1
    return one_hot_y

def forward_prop(x,w1,b1,w2,b2):
    z1 = w1@x + b1
    a1 = ReLU(z1)
    z2 = w2 @ a1 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def predict(output):
    return np.argmax(output)

def test(image, output, trainig_input):
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.title('Prediction: '+str(predict(output))+', Actual: '+str(np.argmax(trainig_input)))
    plt.show()

def train():
    # update weights and biases according to delta rule
    raise NotImplementedError
