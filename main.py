import functions as f
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = f.load_data()
    w1,b1,w2,b2 = f.init_params()
    #plt.imshow(x_train[:,0].reshape(28,28), cmap='gray')
    #plt.show()
    z1,a1,z2,output = f.forward_prop(x_train[:,1],w1,b1,w2,b2)
    f.test(x_train[:,1],output,y_train[:,1])
