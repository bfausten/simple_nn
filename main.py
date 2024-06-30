import functions as f
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    # initialize data and parameters
    (x_train, y_train), (x_test, y_test) = f.load_data()
    w1,b1,w2,b2 = f.init_params()

    # train the model
    lr = 0.09
    w1,b1,w2,b2 = f.train(20,x_train,y_train,w1,b1,w2,b2,lr,x_test,y_test)
