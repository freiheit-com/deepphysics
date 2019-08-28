import pandas as pd
from sklearn.model_selection import train_test_split
import time
import imageio
from collections import defaultdict
import tensorflow as tf
import glob
import numpy as np
import gitignored.remap as remap


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def export_dataframe(df, tfrecords_filename, verbose=False):
    start_time = time.time()
    row_count = len(df.index)

    err_count = 0
    err_per_label = defaultdict(int)
    images_exported = 0

    with tf.io.TFRecordWriter(tfrecords_filename) as writer:

        for idx, (row_idx, row) in enumerate(df.iterrows()):

            if idx % 2500 == 0:
                elapsed_time = time.time() - start_time
                progress_percentage = idx / row_count * 100
                print("progress {}/{} rows ({:.1f}%, time elapsed: {:.1f} seconds)".format(idx, row_count,
                                                                                           progress_percentage,
                                                                                           elapsed_time))


            img_path = row['file_path']
            label = row['int_label']

            try:
                img = imageio.imread(img_path)
            except FileNotFoundError:
                if verbose:
                    print('image ' + img_path + ' not found')
                err_count += 1
                err_per_label[label] += 1
                continue

            width = img.shape[1]
            height = img.shape[0]
            assert width == height

            with open(img_path, mode='rb') as file:  # b is important -> binary
                img_raw = file.read()

            record = tf.train.Example(
                features=tf.train.Features(feature={
                    'image_raw': bytes_feature(img_raw),
                    'int_label': int64_feature(label),
                }))
            writer.write(record.SerializeToString())
            images_exported += 1

        print('Exported {} images. Failed to export {} images'.format(images_exported, err_count))

        
def parse_tfrecord(record):
    parsed = tf.io.parse_single_example(record, {
        'image_raw': tf.io.FixedLenFeature((), tf.string),
        'int_label': tf.io.FixedLenFeature((), tf.int64),
    })
    image = tf.image.decode_png(parsed['image_raw'], channels=1)  # output as grayscale
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, parsed["int_label"]


def load_classification(path_to_classification):
    df = pd.read_csv(path_to_classification, names=["file_name", "label"], delimiter="\t", header=None, dtype=str)
    df["label"] = df["label"].astype(int)

    df.insert(len(df.columns), "generated", False)

    measurement_series = []
    for file_name in df["file_name"]:
        measurement_series.append(
            "{}_{}_{}".format(file_name.split("_")[0], file_name.split("_")[1], file_name.split("_")[2]))

    measurement_index = []
    for file_name in df["file_name"]:
        measurement_index.append(file_name.split("_")[-1].split(".")[0])

    df["measurement_series"] = measurement_series
    df["measurement_index"] = measurement_index
    df["measurement_index"] = df["measurement_index"].astype(int)

    return df


def split_data_set_in_train_and_eval(df, n_examples_per_class):
    dfs_train, dfs_eval = [], []

    for label in df['int_label'].unique():
        df_label = df[df['int_label'] == label]
        df_train, df_eval = train_test_split(df_label,
                                             random_state=42,
                                             shuffle=True,
                                             test_size=n_examples_per_class)
        dfs_train.append(df_train)
        dfs_eval.append(df_eval)

    df_train = pd.concat(dfs_train).sample(frac=1)
    df_eval = pd.concat(dfs_eval).sample(frac=1)

    del dfs_train
    del dfs_eval

    return df_train.sort_index(), df_eval.sort_index()

def gen_filenames(content_of_inputfolder):
    for filename in content_of_inputfolder:
        yield filename
        
def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)  # output as grayscale
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    return image, 0.0

def get_class_weights_map(df):
    class_weights_map = {}
    for i in df.index:
        class_weights_map[i] = np.sum(df["file_path"])/df["file_path"][i]
    return class_weights_map

def get_classification_df(classification_files):
    dfs = []
    for classification_file in classification_files:
        df_buf = pd.read_csv(classification_file, names=["file_path", "ext_label", "ignore"], delimiter="\t", header=None, dtype=str)
        # get folder name for classification file
        folder_name = classification_file.split(".")[0]
        file_path_buf = []
        for file_path in df_buf.file_path:
            file_path_buf.append("{}/{}".format(folder_name, file_path))
        df_buf.file_path = file_path_buf
        df_buf = df_buf.drop(columns=["ignore"])
        df_buf.ext_label = df_buf.ext_label.astype("int")

        dfs.append(df_buf)
        
    df = pd.concat(dfs, ignore_index=True)
    df["int_label"] = remap.remap_cross(df.ext_label)
        
    return df
    