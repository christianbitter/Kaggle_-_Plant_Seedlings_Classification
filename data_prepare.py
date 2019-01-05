import sys
import glob
import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pprint
import tensorflow as tf
import csv
import math
import argparse
from common.common import _bytes_feature, _int64_feature, get_image_binary

# we need to split, such that augmentation puts augmented images into the same input structure as the source data
# with a flag if we want to move input to it as well

def get_labels():
    return {
        "Black-grass": 0,
        "Common Chickweed": 1,
        "Loose Silky-bent": 2,
        "Shepherds Purse": 3,
        "Charlock": 4,
        "Common wheat": 5,
        "Maize": 6,
        "Small-flowered Cranesbill": 7,
        "Cleavers": 8,
        "Fat Hen": 9,
        "Scentless Mayweed": 10,
        "Sugar beet": 11
}


def labelname_to_id(lbl_name:str) -> int:
    return get_labels().get(lbl_name, -1)


def dir_to_class(dir_name: str, verbose: bool = False) -> str:
    if verbose:
        print("dir_name: {0}".format(dir_name))

    return {
        "Black-grass": "Black-grass",
        "Common Chickweed": "Common Chickweed",
        "Loose Silky-bent": "Loose Silky-bent",
        "Shepherds Purse": "Shepherds Purse",
        "Charlock": "Charlock",
        "Common wheat": "Common wheat",
        "Maize": "Maize",
        "Small-flowered Cranesbill": "Small-flowered Cranesbill",
        "Cleavers": "Cleavers",
        "Fat Hen": "Fat Hen",
        "Scentless Mayweed": "Scentless Mayweed",
        "Sugar beet": "Sugar beet"
    }.get(dir_name, "unclassified")


def persist_labels(labels:dict, label_csv_fp:str, verbose: bool=False)->None:
    if not labels:
        raise ValueError("persist_labels - no labels provided")
    if not label_csv_fp:
        raise ValueError("persist_labels - no csv path provided")

    with open(label_csv_fp, 'w', newline='') as csvfile:
        fieldnames = ['label', 'id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for a_label in labels:
            a_label_id = labels[a_label]
            writer.writerow({'label': a_label, 'id': a_label_id})


def prepare(in_dir_fp: str, out_dir_fp: str, opts: dict, preprocessing_fn, img_ext: str =".png",
            verbose: bool=False):
    """
    we separate the raw input images into a structure that it is better suited to
    build class-based training and testing image sets.
    :param in_dir_fp: the input directory, containing the raw images
    :param opts: options to pass
    :param out_dir_fp: the output directory, organized according to classes
    :param verbose: whether or not we want to print diagnostics
    :param preprocessing_fn: a function that takes the path to the image and processes it according to downstream needs
    :return:
    """
    print("prepare: {0}".format(in_dir_fp))

    # we take all the images and move them into a simpler class hierarchy
    files = []
    classes = []

    for fp in glob.iglob("{0}/**/*{1}".format(in_dir_fp, img_ext), recursive=True):
        dir_name = os.path.basename(os.path.dirname(fp)).replace(" ", "_")
        img_name = os.path.basename(fp)
        class_name = dir_to_class(os.path.basename(os.path.dirname(fp)))
        class_dir_fp = "{0}/{1}".format(out_dir_fp, class_name)
        if not os.path.exists(class_dir_fp):
            os.makedirs(class_dir_fp)

        out_fp   = "{0}/{1}_{2}".format(class_dir_fp, dir_name, img_name)
        shutil.copyfile(fp, out_fp)

        _ = preprocessing_fn(out_fp, opts, verbose)
        files.append(out_fp)
        classes.append(class_name)

    print("/prepare: {0}".format(out_dir_fp))

    return files, classes


def split_train_test(in_files:list, in_labels: list, out_dir_fp: str, opts: dict, train_postfix: str="train", test_postfix: str="test",
                     train_pct:float =.95, verbose: bool=False) -> (str, str):
    unique_labels = {l for l in in_labels}
    if verbose:
        print(unique_labels)
    # do a unstratified simple sampling
    X_train, X_test, y_train, y_test = train_test_split(in_files, in_labels, train_size=train_pct, shuffle=True)

    # separate the training data
    out_train_dir_fp = "{0}/{1}".format(out_dir_fp, train_postfix)
    out_test_dir_fp = "{0}/{1}".format(out_dir_fp, test_postfix)
    if not os.path.exists(out_train_dir_fp):
        os.makedirs(out_train_dir_fp)
    if not os.path.exists(out_test_dir_fp):
        os.makedirs(out_test_dir_fp)
    # persist the training file names
    with open(os.path.join(out_train_dir_fp, "training_files.csv"), 'w', newline='') as csvfile:
        fieldnames = ['file', 'class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, x_name in enumerate(X_train):
            a_label_name = y_train[i]
            writer.writerow({'file': x_name, 'class': a_label_name})

    # persist the testing file names
    with open(os.path.join(out_test_dir_fp, "testing_files.csv"), 'w', newline='') as csvfile:
        fieldnames = ['file', 'class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, x_name in enumerate(X_test):
            a_label_name = y_test[i]
            writer.writerow({'file': x_name, 'class': a_label_name})

    for fp in X_train:
        fdst_fp = os.path.join(out_train_dir_fp, os.path.basename(fp))
        shutil.copy(src=fp, dst=fdst_fp)

    for fp in X_test:
        fdst_fp = os.path.join(out_test_dir_fp, os.path.basename(fp))
        shutil.copy(src=fp, dst=fdst_fp)

    return X_train, X_test, y_train, y_test


def tf_record(img_fp: list, lbl: list, tfrecord_fp: str, opts: dict, verbose: bool=False):
    print("Writing TFRecord: {0}".format(tfrecord_fp))

    writer = tf.python_io.TFRecordWriter(tfrecord_fp)

    for idx, f in enumerate(img_fp):
        img_name = os.path.basename(f)
        lbl_name = lbl[idx]

        if verbose:
            print("tf_record/Processing Img/Label: {0}/{1}".format(img_name, lbl_name))

        img_shape_bytes, image_bytes = get_image_binary(fp=f)
        a_feature = {
            'image_shape': _bytes_feature(img_shape_bytes),
            'image': _bytes_feature(image_bytes),
            'label': _int64_feature(labelname_to_id(lbl_name))
        }

        example = tf.train.Example(features=tf.train.Features(feature=a_feature))
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()
    print("/Writing TFRecord: {0}".format(tfrecord_fp))


def augment(images: list, classes: list, augmentation_dir_fp: str, opts: dict,
            aug_opts: dict=None, verbose:bool=False):
    if not images:
        raise ValueError("augment - image dir not provided")
    if not augmentation_dir_fp:
        raise ValueError("augment - augmentation_dir_fp not provided")

    no_images = len(images)

    print("augment - {0} images: {1}".format(no_images, augmentation_dir_fp))

    shift_x, shift_y, rotation_angle_range = aug_opts.get("shift_x", 0), aug_opts.get("shift_y"), aug_opts.get("rotation_angle_range", 0)
    flip_hor, flip_ver = aug_opts.get("flip_hor", False), aug_opts.get("flip_ver", False)
    no_aug_passes = aug_opts.get("no_aug_passes", 1)

    save_format = opts["save_format"]
    aug_prefix = opts["aug_prefix"]
    batch_size = opts["batch_size"]
    # stuff everything into numpy array
    image_width, image_height, no_channels = opts["image_width"], opts["image_height"], opts["no_channels"]
    np_images = np.zeros(shape=(no_images, image_width, image_height, no_channels), dtype=np.uint8)

    vAugmentedClasses = []
    vAugmentedImages = []

    if aug_opts:
        for i, fp_img in enumerate(images):
            im = Image.open(fp_img)
            np_images[i, :, :, :] = np.array(im).reshape(1, image_height, image_width, no_channels)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=rotation_angle_range,
                                                                  width_shift_range=shift_x, height_shift_range=shift_y,
                                                                  horizontal_flip=flip_hor, vertical_flip=flip_ver,
                                                                  data_format="channels_last")
        datagen.fit(np_images, augment=True, rounds=1)
        print("Starting augmentation of images ({0}) ...".format(np_images.shape))
        print("image width : {}".format(image_width))
        print("image height: {}".format(image_height))
        print("no channels : {}".format(no_channels))
        print("format: {}".format(save_format))
        print("prefix: {}".format(aug_prefix))
        print("batch : {}".format(batch_size))
        pprint.pprint(aug_opts)

        j = 0
        j1= 0
        # flow will never return, so we need to stop after the respective number of passes
        iter_stop = math.ceil(no_images / batch_size) * no_aug_passes
        f = datagen.flow(np_images, classes, batch_size=batch_size)
                         # save_to_dir=augmentation_dir_fp, save_prefix=aug_prefix,
                         # save_format=save_format)
        for i in range(iter_stop):
            X_batch, y_batch = f.next()

            # the augmented images are saved and returned in X_batch, y_batch ... so we can add them to the list
            # TODO: change the visualization to show the unmodified top and augmented bottom
            # for i in range(0, 9):
            #     plt.subplot(330 + 1 + i)
            #     img = X_batch[i].reshape(image_height, image_width, no_channels)
            #     plt.imshow(img)
            # plt.show()
            # break

            j1 = j + len(y_batch)
            print("Augmenting Batch...: {0}:{1}/{2} [{3}]".format(j, j1, np_images.shape[0], i))
            fp_images        = images[j:j1]
            augmented_images = ["{0}/{1}_{2}".format(augmentation_dir_fp, aug_prefix, os.path.basename(fp_img)) for fp_img in fp_images]
            # write the image
            if (len(augmented_images) > 0):
                for ix, fp_img in enumerate(augmented_images):
                    image_data = X_batch[ix, ].astype(np.uint8)
                    im = Image.fromarray(image_data)
                    im.save(fp_img)

                vAugmentedClasses.extend(y_batch)
                vAugmentedImages.extend(augmented_images)
            j = j1

        print("Done augmentation of images ...".format(np_images.shape))

    return vAugmentedImages, vAugmentedClasses


def preprocess_image(image_path_fp: str, opts: dict, verbose: bool=False):
    if not (image_path_fp and os.path.exists(image_path_fp)):
        raise ValueError("preprocess_image - image path not provided")

    image_width, image_height = opts["image_width"], opts["image_height"]

    im = Image.open(image_path_fp)
    im = im.convert("RGB")

    if verbose:
        print("preprocess_image: ({0}, {1}, {2})".format(im.format, im.size, im.mode))

    # ideally we have a better way of resizing but for now ...
    out = im
    if image_width > 0 and image_height > 0:
        out = im.resize((image_width, image_height))

    out.save(image_path_fp)
    return image_path_fp


def cli():
    description = "The Fruit360 Data Preparation tool allows you to "
    parser = argparse.ArgumentParser(prog="Data Preparation",
                                     description=description)
    parser.add_argument("--input-dir", type=str, required=False, help="The full path to the directory containing raw input images.", dest="input_dir")
    parser.add_argument("--prepared-dir", type=str, required=False, help="The full path to the directory where augmented images should be stored.", dest="prep_dir")
    parser.add_argument("--augmented-dir", type=str, required=False, help="The full path to the directory where augmented images should be stored.", dest="aug_dir")
    parser.add_argument("--split-dir", type=str, required=False, help="The full path to the directory where augmented images should be stored.", dest="split_dir")
    parser.add_argument("--tfrecords-dir", type=str, required=False, help="The full path to the directory where augmented images should be stored.", dest="tf_dir")
    parser.add_argument("--label-dir", type=str, required=False, help="The full path to the directory where augmented images should be stored.", dest="label_dir")
    return parser


def main():

    # arg_parser = cli()
    # parsed = arg_parser.parse_args()
    base_data_dir = "E:/Data/Kaggle_-_Plant_Seedlings_Classification"
    prefix = "small_"
    # prefix = ""
    img_fp_train, img_fp_test, lbl_train, lbl_test = None, None, None, None
    do_augment = True
    do_split   = True
    do_tfrecord= True
    verbose    = False

    opts = {
        "image_width": 100,
        "image_height": 100,
        "no_channels": 3,
        "save_format": "png",
        "aug_prefix": "aug",
        "batch_size": 64
    }
    aug_opts = {
        "flip_hor": True,
        "flip_ver": True,
        "no_aug_passes": 2,
    }

    print("Data Preparation:")
    print("Augment Images: {}".format(do_augment))
    print("Split to Train/ Test: {}".format(do_split))
    print("Create TFRecords: {}".format(do_tfrecord))

    directories = {
        "input_dir_fp": "{0}/{1}train".format(base_data_dir, prefix),
        "normalized_dir_fp": "{0}/{1}normalized".format(base_data_dir, prefix),
        "prepared_dir_fp": "{0}/{1}prepared".format(base_data_dir, prefix),
        "split_dir_fp": "{0}/{1}split".format(base_data_dir, prefix),
        "tf_dir_fp": "{0}/{1}tf_data".format(base_data_dir, prefix),
        "lbl_dir_fp": "{0}/{1}label".format(base_data_dir, prefix),
        "aug_dir_fp": "{0}/{1}augmented".format(base_data_dir, prefix)
    }

    print("Creating output directories ...:")
    for k in directories:
        if not os.path.exists(directories[k]):
            os.makedirs(directories[k])

    files, classes = prepare(in_dir_fp=directories["input_dir_fp"],
                             out_dir_fp=directories["prepared_dir_fp"],
                             opts=opts,
                             preprocessing_fn=preprocess_image)

    print("Organized: {0} files/ {1} classes".format(len(files), len(set(classes))))

    if do_augment:
        augmented_files, augmented_classes = augment(files, augmentation_dir_fp=directories["aug_dir_fp"],
                                                     opts=opts, aug_opts=aug_opts,
                                                     classes=classes, verbose=verbose)
        if augmented_files and len(augmented_files) > 0:
            files.extend(augmented_files)
            classes.extend(augmented_classes)
        print("Total: {0} files/ {1} classes".format(len(files), len(set(classes))))

    if do_split:
        img_fp_train, img_fp_test, lbl_train, lbl_test = split_train_test(in_files=files,
                                                                          in_labels=classes,
                                                                          out_dir_fp=directories["split_dir_fp"],
                                                                          train_postfix="train",
                                                                          test_postfix="test",
                                                                          train_pct=.95,
                                                                          opts=opts,
                                                                          verbose=verbose)
        print("Split into Testing and Training ...")

    if do_tfrecord:
        tf_record(img_fp=img_fp_train, lbl=lbl_train, opts=opts,
                  tfrecord_fp=os.path.join(directories["tf_dir_fp"], "train.rec"), verbose=verbose)
        print("TF Records written for training")
        tf_record(img_fp=img_fp_test, lbl=lbl_test, opts=opts,
                  tfrecord_fp=os.path.join(directories["tf_dir_fp"], "test.rec"), verbose=verbose)
        print("TF Records written for testing")

    persist_labels(labels=get_labels(), label_csv_fp=os.path.join(directories["lbl_dir_fp"], "label_dict.csv"), verbose=verbose)


if __name__ == '__main__':
    main()