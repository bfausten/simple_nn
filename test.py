import functions as f
import main as m
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# show 20 random images from the test set and their predictions in the same plot
(x_train, y_train), (x_test, y_test) = f.load_data()
w1,b1,w2,b2,preds = np.load('w1.npy'), np.load('b1.npy'), np.load('w2.npy'), np.load('b2.npy'), np.load('predictions.npy')

def show_random_predictions(x_test, y_test, w1, b1, w2, b2, num_images=20):
    indices = np.random.choice(x_test.shape[1], num_images, replace=False)
    fig, axes = plt.subplots(2, 10, figsize=(13.5, 4))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        _, _, _,a2 = f.forward_prop(x_test[:, idx], w1, b1, w2, b2)
        prediction = f.predict(a2)
        actual = np.argmax(y_test[:, idx])
        axes[i].imshow(x_test[:, idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Pred: {prediction}, Act: {actual}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def test_image(image):
    _, _, _,a2 = f.forward_prop(image, w1, b1, w2, b2)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Prediction: {f.predict(a2)}')
    plt.show()

def save_predictions(x_test):
    # save the predictions to a file
    w1,b1,w2,b2 = np.load('w1.npy'), np.load('b1.npy'), np.load('w2.npy'), np.load('b2.npy')
    predictions = []
    for i in range(x_test.shape[1]):
        _, _, _,a2 = f.forward_prop(x_test[:, i], w1, b1, w2, b2)
        predictions.append(f.predict(a2))
    np.save('predictions.npy', predictions)

def false_positives_of_labels(y_test):
    # calculate the false positive rate for each label
    for i in range(10):
        indices = np.where(preds == i)[0]
        false_positives = np.sum(i != np.argmax(y_test[:, indices], axis=0))
        fp_rate = false_positives/len(indices) * 100
        print(f'False positive rate of label {i}:', fp_rate, '%')
        print(f'Number of false positives:', false_positives, 'out of', len(indices))
        print('\n')
