import sys
# sys.path.append("/opt/anaconda3/envs/tf/lib/python3.7/site-packages")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import classification_report

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
report = open("figures/classification_report.txt", "w")


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    'save a figure with fig_id'
    path = os.path.join("figures", fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def loadDatasets(dataset):
    'loads the data set, prints sets shape, plots some examples'
    # load fashion mnist dataset and split it into train, test and validation set
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    # print train, test and validation shape
    report.write("X_train_full shape = " + str(X_train_full.shape) + "\n")
    report.write("y_train_full shape = " + str(y_train_full.shape) + "\n")
    report.write("X_test shape = " + str(X_test.shape) + "\n")
    report.write("y_test shape = " + str(y_test.shape) + "\n")
    # scale the pixel intensities down to the 0–1 range by dividing them by 255.0
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.
    report.write("X_valid shape = " + str(X_valid.shape) + "\n")
    report.write("y_valid shape = " + str(y_valid.shape) + "\n")
    report.write("X_train shape = " + str(X_train.shape) + "\n")
    report.write("y_train shape = " + str(y_train.shape) + "\n")
    # plotting the first 40 items with the name of their classes
    plotItemsWithLabels(X_train, y_train)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def plotLearningCurve(history, optimizer):
    'plotting the learning curve'
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    if optimizer is None:
        plt.title(label="keras_learning_curves_plot_with_rmsp_optimizer")
        save_fig("keras_learning_curves_plot_with_rmsp_optimizer")
    else:
        plt.title(label="keras_learning_curves_plot_with_" + optimizer + "_optimizer")
        save_fig("keras_learning_curves_plot_with_" + optimizer + "_optimizer")
    plt.close("all")


def plotItemsWithLabels(X, y, model=None, n_rows=4, n_cols=10, optimizer=None):
    'plots items with their label (+prediction)'
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    if model is not None:
        y_prediction_classes = model.predict_classes(X)
        if optimizer is None:
            id = "fashion_mnist_plot_with_labels_rmsp"
        else:
            id = 'fashion_mnist_plot_with_labels_' + optimizer
    else:
        id = 'fashion_mnist_plot_with_labels'
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            if model is not None:
                plt.title(class_names[y[index]] + " p:" + class_names[int(y_prediction_classes[index])], fontsize=4)
            else:
                plt.title(class_names[y[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    # plt.title(label=id)
    save_fig(id, tight_layout=False)
    plt.close("all")


def ANN(sets, loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=None, epochs=30):
    'builds, compiles and trains a NN'
    if optimizer == 'momentum':
        optimizer = keras.optimizers.SGD(momentum=0.9)
        optimizerName = "momentum"
    else:
        optimizerName = optimizer
    X_train, y_train, X_valid, y_valid, X_test, y_test = sets
    # building a classification MLP with two hidden layers
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.summary()
    # compiling the model (if optimizers are used, hyper parameters are their default value)
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer if (optimizer) else "rmsprop")
    # training the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid))
    # plot learning curve
    plotLearningCurve(history, optimizerName)
    # make prediction on test set
    y_prediction = model.predict(X_test)
    y_prediction_bool = np.argmax(y_prediction, axis=1)
    plotItemsWithLabels(X_test, y_test, model=model, optimizer=optimizerName)
    # print the classification report
    report.write("   -------------------------------------------------\n")
    if optimizerName is None:
        report.write("classification report with rmsp optimizer" +  "\n")
    else:
        report.write("classification report with "+ optimizerName +  "\n")
    report.write(classification_report(y_test, y_prediction_bool) + "\n")

if __name__ == "__main__":
    np.random.seed(42)
    fashion_mnist = keras.datasets.fashion_mnist
    sets = loadDatasets(fashion_mnist)
    ANN(sets, optimizer="sgd")
    ANN(sets, optimizer="momentum")
    ANN(sets)
    ANN(sets, optimizer="adam")
