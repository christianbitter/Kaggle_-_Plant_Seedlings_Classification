import os
import math
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
from common import ImageClassificationModel
from common.common import vgg_layer


class PlantSeedlingClassModel(ImageClassificationModel):
    def __init__(self, params_fp=None, verbose=False):
        super(PlantSeedlingClassModel, self).__init__(params_fp=params_fp, verbose=verbose)
        self.tfrecord_train_fp = self._load_checked_param("training_tfrecord_fp")
        self.steps_per_epoch   = math.ceil(len(self.training_data) / self.batch_size)

    def _global_preprocess(self, data, label, args=None):
        if data is not None:
            data = tf.cast(data, dtype=tf.float32)
            data = data / 255.
        if label is not None:
            label = tf.cast(label, dtype=tf.int64)
        return data, label

    def import_data(self):
        if self.verbose:
            print("PlantSeedlingClassModel.import_data")

        if not self.tfrecord_train_fp:
            raise ValueError("PlantSeedlingClassModel.import_data - tfrecord_train_fp missing")

        train_dataset = tf.data.TFRecordDataset(self.tfrecord_train_fp)
        train_dataset = train_dataset.map(self._tfrecord_parse_function, num_parallel_calls=8)
        train_dataset = train_dataset.map(self._global_preprocess, num_parallel_calls=8)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(self.shuffle_buffer)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(self.batch_size)
        iterator = train_dataset.make_one_shot_iterator()
        #
        image, label = iterator.get_next()
        self.data_iter = tf.reshape(image, [-1, self.image_width, self.image_height, self.no_image_channels])
        self.label_iter = tf.one_hot(label, len(self.classes()))
        self.training_set = train_dataset

    def build(self):
        use_input_dropout = False
        use_layer_dropout = False
        layer_dropout_rate = 0.
        input_dropout_rate = 0.

        if 'input_dropout' in self.params:
            use_input_dropout = True
            input_dropout_rate = self.params['input_dropout']
            if input_dropout_rate <= 0:
                use_input_dropout = False

        if 'layer_dropout' in self.params:
            use_layer_dropout = True
            layer_dropout_rate = self.params['layer_dropout']
            if layer_dropout_rate <= 0:
                use_layer_dropout = False

        with tf.name_scope('input'):
            self.model_input = tf.keras.layers.Input(shape=(self.image_width, self.image_height, self.no_image_channels))
            x = self.model_input
            if use_input_dropout:
                x = keras.layers.Dropout(rate=input_dropout_rate)(x)

        with tf.name_scope('layer1'):
            x = vgg_layer(filter_size=32, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope('layer2'):
            x = vgg_layer(filter_size=64, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope('layer3'):
            x = vgg_layer(filter_size=256, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope('layer4'):
            x = vgg_layer(filter_size=512, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope('layer5'):
            x = vgg_layer(filter_size=1024, input=x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

        with tf.name_scope("output"):
            x = keras.layers.Flatten()(x)

            x = keras.layers.Dense(1024)(x)
            if use_layer_dropout:
                x = keras.layers.Dropout(rate=layer_dropout_rate)(x)

            model_output = tf.keras.layers.Dense(len(self.classes()), activation='softmax')(x)
            self.model_output = model_output

    def preprocess_input(self, image: Image):
        # TODO: this has to incorporate the same code as the data set preprocessing
        x = np.asarray(image, np.uint8)
        x = np.array(x).reshape((1, self.image_height, self.image_width, self.no_image_channels))

        x, _ = self._global_preprocess(x, None)
        return x

    def _get_checkpoint_filename(self):
        return "{0}_{1}_{2}_{3}_{4}.ckpt".format(self.model_name, self.learning_rate, self.batch_size, self.no_epochs, "{epoch:04d}")

    def plot(self, what: str=None):
        pass

    def summary(self, what: str='full', to_console: bool = False) -> str:
        s = self.model.summary()
        if to_console:
            print(s)

        return s