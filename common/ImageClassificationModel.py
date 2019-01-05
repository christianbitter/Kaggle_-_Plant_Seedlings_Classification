import csv
import math
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from .Model import Model

# TODO: real model evaluation with predict, so that we can generate the confusion matrix, ROC, etc.

class ImageClassificationModel(Model):
    def __init__(self, params_fp, verbose=False):
        super(ImageClassificationModel, self).__init__(params_fp=params_fp, verbose=verbose)
        self.data_iter = None
        self.label_iter = None
        self.steps_per_epoch = None
        self.model_input = None
        self.model_output = None
        self.model = None
        self.class_ids = dict()
        self.training_data = dict()

        self.batch_size = self._load_checked_param("batch_size")
        self.no_epochs = self._load_checked_param("no_epochs")
        self.learning_rate = self._load_checked_param("learning_rate")
        self.optimizer_name = self._load_checked_param("optimizer")
        self.training_set = self._load_checked_param("batch_size")
        self.use_tensorboard = self._load_checked_param("use_tensorboard")
        self.checkpoint_dir_fp = self._load_checked_param("checkpoint_fp")
        self.logging_dir_fp = self._load_checked_param("log_fp")

        self.image_width = self._load_checked_param("image_width")
        self.image_height = self._load_checked_param("image_height")
        self.no_image_channels = self._load_checked_param("image_channels")
        self.labels_fp = self._load_checked_param("labels_fp")
        self.training_fp = self._load_checked_param("training_fp")
        self.do_checkpointing = self._load_checked_param("checkpoint") != 0

        self.shuffle_buffer = self.batch_size

        self._load_classes()
        self._load_training_data()


    def _load_training_data(self):
        if not (self.training_fp and os.path.exists(self.training_fp)):
            raise ValueError("ImageClassificationModel._load_training_data - no data file provided")

        with open(self.training_fp) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            for row in reader:
                self.training_data[row['file']] = row['class']

        if not self.training_data:
            raise ValueError("ImageClassificationModel._load_training_data - no data provided")

    def _load_classes(self):
        if not (self.labels_fp and os.path.exists(self.labels_fp)):
            raise ValueError("ImageClassificationModel._load_classes - no class file provided")

        with open(self.labels_fp) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            for row in reader:
                if self.verbose:
                    print("Adding to labels: {0} => {1}".format(row['label'], int(row['id'])))
                k = row['label'].lower()
                v = int(row['id'])
                self.class_ids[k] = v

        assert (isinstance(self.class_ids, dict))

        if not self.class_ids:
            raise ValueError("ImageClassificationModel._load_classes - no classes provided")

    def _global_preprocess(self, data, label, **kwargs):
        return data, label

    def _parse_tfrecord(self, tfrecord_proto):
        raise ValueError("ImageClassificationModel._parse_tfrecord - not implemented")

    def _get_checkpoint_filename(self):
        raise ValueError("ImageClassificationModel._get_checkpoint_filename - not implemented")

    def import_data(self):
        raise ValueError("import_data - not implemented")

    def train(self):
        if self.data_iter is None or self.label_iter is None:
            raise ValueError("ImageClassificationModel.train - data iterator or label iterator not defined")

        callbacks = []
        if self.use_tensorboard and self.logging_dir_fp:
            if self.verbose:
                print("Enabling Keras Tensorboard Callback: {0}".format(self.logging_dir_fp))
            kcb = tf.keras.callbacks.TensorBoard(log_dir=self.logging_dir_fp, histogram_freq=0, write_graph=True, write_images=False)
            callbacks.append(kcb)

        if self.do_checkpointing and self.checkpoint_dir_fp:
            save_best = False
            checkpoint_path = "{0}/{1}".format(self.checkpoint_dir_fp, self._get_checkpoint_filename())
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=save_best, verbose=1)
            callbacks.append(cp_callback)

        return self.model.fit(
            self.data_iter, self.label_iter,
            epochs=self.no_epochs, steps_per_epoch=self.steps_per_epoch, verbose=True, callbacks=callbacks)

    def summary(self, what: str = 'full', to_console: bool = False) -> str:
        raise ValueError("ImageClassificationModel.summary - not implemented")

    def to_checkpoint(self, checkpoint_name):
        if not self.model:
            raise ValueError("ImageClassificationModel.to_checkpoint - cannot persist model to checkpoint (not trained?)")

        checkpoint_fp = "{0}/{1}"
        checkpoint_dir_fp = "checkpoint"
        if self.checkpoint_dir_fp:
            checkpoint_dir_fp = self.checkpoint_dir_fp

        checkpoint_fp = checkpoint_fp.format(checkpoint_dir_fp, checkpoint_name)
        self.model.save(checkpoint_fp, overwrite=True, include_optimizer=True)

        return checkpoint_fp

    def from_checkpoint(self, checkpoint_name: str, full_checkpoint: bool=False):
        if self.verbose:
            print("ImageClassificationModel.from_checkpoint - loading model from checkpoint")

        checkpoint_fp = "{0}/{1}"
        checkpoint_dir_fp = "checkpoint"
        if self.checkpoint_dir_fp:
            checkpoint_dir_fp = self.checkpoint_dir_fp
        checkpoint_fp = checkpoint_fp.format(checkpoint_dir_fp, checkpoint_name)

        if full_checkpoint:
            self.model = tf.keras.models.load_model(checkpoint_fp)
        else:
            self.build()
            self.prepare(model_weights_fp=checkpoint_fp)

    def predict(self, image: Image) -> dict:
        if not image:
            raise ValueError("ImageClassificationModel.predict - image not provided")
        if not self.model:
            raise ValueError("ImageClassificationModel.predict - cannot predict on untrained model")
        x = self.preprocess_input(image)
        s = self.model.predict(x, steps=1, verbose=True)
        ci = np.argmax(s)

        return {
            "class_id": ci,
            "probability": np.max(s),
            "class_name": self.classid_to_class(ci),
            "predictions": s
        }

    # todo: we need an iterator, where we pass a list of files and classes and resolve to images
    # todo: the generator needs to go through the same preprocessing functions
    def predict_from_directory(self, input_dir_fp):
        image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        # image_datagen.fit(None, augment=True, seed=seed)
        image_generator = image_datagen.flow_from_directory(input_dir_fp,
                                                            target_size=(self.image_width, self.image_height),
                                                            color_mode="rgb",
                                                            class_mode="categorical",
                                                            seed=seed)
        s = self.model.predict_generator(image_generator, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=self.verbose)

        predicted_class_indices = np.argmax(s, axis=1)
        actual_class_name = [os.path.dirname(fn).lower() for fn in image_generator.filenames]
        actual_class_indices = [self.class_ids[cn] for cn in actual_class_name]
        predicted_class_name = [self.classid_to_class(i).lower() for i in predicted_class_indices]

        return {
            "predicted_class_id": predicted_class_indices,
            "predicted_class_name": predicted_class_name,
            "actual_class_id": actual_class_indices,
            "actual_class_name": actual_class_name,
            "class_labels": image_generator.class_indices,
            "file_names" : image_generator.filenames,
            "predictions": s
        }

    def prepare(self, model_weights_fp: str=None):
        self.model = tf.keras.models.Model(inputs=self.model_input, outputs=self.model_output)
        optimizer = {
            "rmsprop": tf.keras.optimizers.RMSprop(lr=self.learning_rate),
            "adam": tf.keras.optimizers.Adam(lr=self.learning_rate)
        }.get(self.optimizer_name, None)
        if not optimizer:
            raise ValueError("Model - unknown optimizer '{0}'.".format(self.optimizer_name))

        if model_weights_fp:
            weights_fp_idx = "{0}.index".format(model_weights_fp)

            if os.path.exists(weights_fp_idx):
                self.model.load_weights(model_weights_fp)
            else:
                raise ValueError("Model - weights file specified but does not exist")

        if 'loss_function' in self.params:
            fn_loss_name = self.params['loss_function']
        else:
            fn_loss_name = 'mean_squared_error'

        if 'performance_metric' in self.params:
            v_metrics = self.params['performance_metric']
        else:
            v_metrics = ['acc', 'mae']

        self.model.compile(optimizer=optimizer, loss=fn_loss_name, metrics=v_metrics)

    def _tfrecord_parse_function(self, tfrecord_proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        features = {'label': tf.FixedLenFeature([], tf.int64),
                    'image_shape': tf.FixedLenFeature([], tf.string),
                    'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(tfrecord_proto, features)

        label = parsed_features['label']
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        shape = tf.decode_raw(parsed_features['image_shape'], tf.uint8)
        return image, label, shape

    def evaluate(self, tfrecord_test_fp: str=None, test_label_fp: str=None):
        # if nothing is provided load from params
        if not tfrecord_test_fp:
            tfrecord_test_fp = self._load_checked_param("testing_tfrecord_fp")
        if not test_label_fp:
            test_label_fp = self._load_checked_param("testing_fp")

        if not (tfrecord_test_fp and os.path.exists(tfrecord_test_fp)):
            raise ("ImageClassificationModel.evaluate - tfrecord_test_fp cannot be null and has to exist")
        if not (test_label_fp and os.path.exists(test_label_fp)):
            raise ("ImageClassificationModel.evaluate - test_label_fp cannot be null and has to exist")

        testing_data = {}
        with open(test_label_fp) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
            for row in reader:
                testing_data[row['file']] = row['class']

        if not testing_data:
            raise ValueError("ImageClassificationModel.evaluate - no testing data provided")

        if self.verbose:
            print("tfrecord_test_fp: {0}".format(tfrecord_test_fp))
            print("test_label_fp   : {0} ({1} Instances)".format(test_label_fp, len(testing_data)))

        iterator  = self._tfrecord_to_iter(tfrecord_fp=tfrecord_test_fp, do_batch=True, do_prefetch=True)
        image, label = iterator.get_next()
        image_iter = tf.reshape(image, [-1, self.image_width, self.image_height, self.no_image_channels])
        label_iter = tf.one_hot(label, len(self.classes()))

        no_testing_steps = math.ceil(len(testing_data) / self.batch_size)

        return self.model.evaluate(x=image_iter, y=label_iter,  steps=no_testing_steps, verbose=self.verbose)

    def _tfrecord_to_iter(self, tfrecord_fp: str, **kwargs):
        if not tfrecord_fp:
            raise ValueError("ImageClassificationModel.tfrecord_to_iter - tfrecord file path not provided")
        if not os.path.exists(tfrecord_fp):
            raise ValueError("ImageClassificationModel.tfrecord_to_iter - tfrecord file does not exist")

        dop = 1
        do_batch = False
        do_prefetch = False
        do_shuffle = False
        do_repeat = False

        if 'dop' in kwargs:
            dop = kwargs['dop']
        if 'do_batch' in kwargs:
            do_batch = kwargs['do_batch']
        if 'do_prefetch' in kwargs:
            do_prefetch = kwargs['do_prefetch']
        if 'do_shuffle' in kwargs:
            do_shuffle = kwargs['do_shuffle']
        if 'do_repeat' in kwargs:
            do_repeat = kwargs['do_repeat']

        ds = tf.data.TFRecordDataset(tfrecord_fp)
        ds = ds.map(self._tfrecord_parse_function, num_parallel_calls=dop)
        ds = ds.map(self._global_preprocess, num_parallel_calls=dop)

        if do_repeat:
            ds = ds.repeat()
        if do_shuffle:
            ds = ds.shuffle(self.shuffle_buffer)
        if do_batch:
            ds = ds.batch(self.batch_size)
        if do_prefetch:
            ds = ds.prefetch(self.batch_size)

        return ds.make_one_shot_iterator()

    def class_to_id(self, lbl_name: str) -> int:
        return self.class_ids.get(lbl_name, -1)

    def classid_to_class(self, class_id: int) -> str:
        return [m for m in self.class_ids if int(self.class_ids[m]) == class_id][0]

    def classes(self):
        return [key for key in self.class_ids]

    def preprocess_input(self, image: Image):
        raise ValueError("ImageClassificationModel.preprocess_input - not implemented")