import sys
import glob
import os
import shutil
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import tensorflow as tf
import csv
import random
import argparse
from common.common import _bytes_feature, _int64_feature, get_image_binary


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


def prepare(in_dir_fp: str, out_dir_fp: str, preprocessing_fn, img_ext: str =".png", verbose: bool=False):
    """
    we separate the raw input images into a structure that it is better suited to
    build class-based training and testing image sets.
    :param in_dir_fp: the input directory, containing the raw images
    :param out_dir_fp: the output directory, organized according to classes
    :param verbose: whether or not we want to print diagnostics
    :param preprocessing_fn: a function that takes the path to the image and processes it according to downstream needs
    :return:
    """
    if verbose:
        print("prepare: {0}".format(in_dir_fp))
    # we take all the images and move them into a simpler class hierarchy
    files = []
    classes = []
    for fp in glob.iglob("{0}/**/*{1}".format(in_dir_fp, img_ext), recursive=True):
        if verbose:
            print("Processing: {0}".format(fp))
        dir_name = os.path.basename(os.path.dirname(fp)).replace(" ", "_")
        img_name = os.path.basename(fp)
        class_name = dir_to_class(os.path.basename(os.path.dirname(fp)))
        if verbose:
            print("Class: {0}".format(class_name))
        class_dir_fp = "{0}/{1}".format(out_dir_fp, class_name)
        if not os.path.exists(class_dir_fp):
            os.makedirs(class_dir_fp)

        out_fp   = "{0}/{1}_{2}".format(class_dir_fp, dir_name, img_name)
        shutil.copyfile(fp, out_fp)
        if verbose:
            print("Processing (Pre-processing): {}".format(out_fp))
        _ = preprocessing_fn(out_fp, verbose)
        files.append(out_fp)
        classes.append(class_name)

    return files, classes


def split_train_test(in_files:list, in_labels:list, out_dir_fp: str, train_postfix: str="train", test_postfix: str="test", train_pct:float =.95, verbose: bool=False) -> (str, str):
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


def tf_record(img_fp: list, lbl: list, tfrecord_fp:str, verbose: bool=False):
    if verbose:
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


def augment(images:list, classes:list, augmentation_dir_fp:str, verbose:bool=False):
    if not images:
        raise ValueError("augment - image dir not provided")
    if not augmentation_dir_fp:
        raise ValueError("augment - augmentation_dir_fp not provided")

    if verbose:
        print("augment - {0} images".format(len(images)))

    vAugmentedImages = []
    vAugmentedClasses = []
    for i, image_fp in enumerate(images):
        augmented_images = augment_image(image_fp=image_fp, augmentation_dir_fp=augmentation_dir_fp, verbose=verbose)
        if len(augmented_images) > 0:
            augmented_classes = np.repeat(classes[i], repeats=len(augmented_images))
            vAugmentedImages.extend(augmented_images)
            vAugmentedClasses.extend(augmented_classes)

    return vAugmentedImages, vAugmentedClasses

# todo: switch to keras for augmentation
def augment_image(image_fp, augmentation_dir_fp:str, verbose:bool=False):
    # let's define those probs here now, but take them as parms later
    p_mirror = .5
    p_rotate = .5
    p_scale  = .5
    p_shift  = .5
    f_angle  = random.randint(1, 359)
    # define a random for the number of images ... 0 - 4 ... we want to create
    a_image = Image.open(image_fp)
    n_image = random.randint(0, 4)
    image_path_fp = os.path.dirname(image_fp)
    img_name      = os.path.basename(image_fp)

    augmented_images = []

    for i in range(n_image):
        suffix = ""
        if verbose:
            print("Creating {0} augmented images".format(n_image))

        if random.random() >= p_mirror:
            a_image = ImageOps.mirror(a_image)
            suffix = "_mr"

        if random.random() >= p_rotate:
            a_image = a_image.rotate(f_angle, expand=0)
            suffix += "_rt"

        # if random.random() >= p_scale:
        #     suffix = "_mc"
        #
        # if random.random() >= p_shift:
        #     suffix = "_ms"

        if suffix != "":
            f_name = img_name[0:(len(img_name) - 4)]
            f_ext  = img_name[(len(img_name) - 3):]
            t_image_name = "{0}{1}.{2}".format(f_name, suffix, f_ext)
            image_out_fp = os.path.join(augmentation_dir_fp, t_image_name)
            if verbose:
                print("Writing: {0}".format(image_out_fp))
            a_image.save(image_out_fp)
            augmented_images.append(image_out_fp)

    return augmented_images


def preprocess_image(image_path_fp: str, verbose: bool=False):
    if not (image_path_fp and os.path.exists(image_path_fp)):
        raise ValueError("preprocess_image - image path not provided")

    image_width, image_height = 250, 250

    im = Image.open(image_path_fp)
    if verbose:
        print("preprocess_image: converting to RGB")
    im = im.convert("RGB")

    print("preprocess_image: ({0}, {1}, {2})".format(im.format, im.size, im.mode))

    # ideally we have a better way of resizing but for now ...
    if image_width > 0 and image_height > 0:
        if verbose:
            print("preprocess_image: resizing ({0}, {1})".format(image_width, image_height))

        out = im.resize((image_width, image_height))

    if verbose:
        print("preprocess_image: saving")

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
    input_dir_fp = "{0}/train".format(base_data_dir)
    prepared_dir_fp = "{0}/prepared".format(base_data_dir)
    split_dir_fp    = "{0}/split".format(base_data_dir)
    tf_dir_fp    = "{0}/tf_data".format(base_data_dir)
    lbl_dir_fp   = "{0}/label".format(base_data_dir)
    aug_dir_fp   = "{0}/augmented".format(base_data_dir)
    do_augment = False
    verbose=False

    if not os.path.exists(prepared_dir_fp):
        os.makedirs(prepared_dir_fp)
    if not os.path.exists(split_dir_fp):
        os.makedirs(split_dir_fp)
    if not os.path.exists(tf_dir_fp):
        os.makedirs(tf_dir_fp)
    if not os.path.exists(lbl_dir_fp):
        os.makedirs(lbl_dir_fp)
    if not os.path.exists(aug_dir_fp):
        os.makedirs(aug_dir_fp)
    #
    files, classes = prepare(in_dir_fp=input_dir_fp, out_dir_fp=prepared_dir_fp, preprocessing_fn=preprocess_image, verbose=verbose)
    print("Organized: {0} files/ {1} classes".format(len(files), len(set(classes))))

    if do_augment:
        augmented_files, augmented_classes = augment(files, augmentation_dir_fp=aug_dir_fp, classes=classes, verbose=verbose)
        if augmented_files and len(augmented_files) > 0:
            print("Augmented: {0} files/ {1} classes".format(len(augmented_files), len(augmented_classes)))
            files.extend(augmented_files)
            classes.extend(augmented_classes)
            print(type(augmented_classes))
        print("Total: {0} files/ {1} classes".format(len(files), len(classes)))
    img_fp_train, img_fp_test, lbl_train, lbl_test = split_train_test(in_files=files,
                                                                      in_labels=classes,
                                                                      out_dir_fp=split_dir_fp,
                                                                      train_postfix="train",
                                                                      test_postfix="test",
                                                                      train_pct=.95,
                                                                      verbose=verbose)
    print("Split into Testing and Training")
    tf_record(img_fp=img_fp_train, lbl=lbl_train, tfrecord_fp= os.path.join(tf_dir_fp, "train.rec"), verbose=verbose)
    print("TF Records written for training")
    tf_record(img_fp=img_fp_test, lbl=lbl_test, tfrecord_fp=os.path.join(tf_dir_fp, "test.rec"), verbose=verbose)
    print("TF Records written for testing")
    persist_labels(labels=get_labels(), label_csv_fp=os.path.join(lbl_dir_fp, "label_dict.csv"), verbose=verbose)


if __name__ == '__main__':
    main()