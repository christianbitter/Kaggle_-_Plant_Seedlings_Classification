import numpy as np
from  PIL import Image
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_image_binary(fp:str):
    image = Image.open(fp)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)

    return (shape.tobytes(), image.tobytes())


def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def vgg_layer(filter_size:int, input, padding:str='same', activation:str='relu'):
    x = input
    x = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding=padding)(x)
    x = keras.layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation=activation)(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    return x