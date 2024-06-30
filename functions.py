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

def one_hot(y):
    nrows = 10
    ncols = len(y)
    one_hot_y = np.zeros((nrows,ncols))
    one_hot_y[y,np.arange(ncols)] = 1
    return one_hot_y

def forward_prop(x,w1,b1,w2,b2):
    z1 = w1@x + b1
    a1 = ReLU(z1)
    a2 = w2 @ a1 + b2
    return z1, a1, a2

def predict(output):
    return np.argmax(output, axis=0)

def test_image(image, output, training_input):
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.title('Prediction: '+str(predict(output))+', Actual: '+ str(np.argmax(training_input)))
    plt.show()

def train(epochs, x, y, w1, b1, w2, b2, lr, x_test, y_test):
    # update weights and biases according to delta rule
    for epoch in range(epochs):
        acc_counter = 0
        for i in range(60000):
            z1, a1, a2 = forward_prop(x[:,i], w1, b1, w2, b2)
            if predict(a2) == predict(y[:,i]):
                acc_counter +=1
            # output layer
            da2 = y[:,i] - a2
            dw2 = np.outer(da2, a1)
            db2 = da2
            # hidden layer
            temp = w2.T @ da2
            da1 = ReLU_deriv(z1) * temp
            dw1 = np.outer(da1,x[:,i])
            db1 = temp

            # upadte weights and biases
            w1 -= lr * dw1
            b1 -= lr * db1
            w2 -= lr * dw2
            b2 -= lr * db2
        accuracy = acc_counter / 500
        if epoch % 1 == 0:
            print('Epoch:', epoch+1, ', Training accuracy:', accuracy, '%')
    test_counter=0
    for i in range(10000):
        _,_,test_output= forward_prop(x_test[:,i],w1,b1,w2,b2)
        if predict(test_output) == predict(y_test[:,i]):
            test_counter += 1
    test_accuracy = test_counter / 100
    print('Final Test accuracy:', test_accuracy, '%')
    np.save("w1.npy", w1)
    np.save("b1.npy", b1)
    np.save("w2.npy", w2)
    np.save("b2.npy", b2)
    return w1, b1, w2, b2
