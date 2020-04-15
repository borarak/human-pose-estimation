import tensorflow as tf
import numpy as np
import json
import os
import cv2
from matplotlib import pyplot as plt


def keypoints2xy(keypoints_arr):
    keypoints = []
    for i in range(0, len(keypoints_arr), 3):
        if keypoints_arr[i + 2] == 2:
            keypoints.append((keypoints_arr[i], keypoints_arr[i + 1]))
        else:
            keypoints.append((-1, -1))

    return keypoints


def get_coco_data(data_dir, annotation_file):
    if not os.path.exists(data_dir) or not os.path.isfile(annotation_file):
        raise ValueError("data and annotation path must be provided")
    with open(os.path.join(data_dir, annotation_file), "r") as json_file:
        data = json.load(json_file)
    return data


def get_coco_dataset(data_dir, annotation_file):
    data = get_coco_data(data_dir, annotation_file)
    file_names = []
    bbox = []
    original_keypoints = []

    for elem in data:
        file_names.append(elem['file_name'])
        bbox.append(elem['bbox'])
        original_keypoints.append(elem['keypoints'])

    dataset = tf.data.Dataset.from_tensor_slices(
        (file_names, bbox, original_keypoints))
    dataset = dataset.map(lambda x, y, z: (tf.image.decode_jpeg(
        tf.io.read_file(x), channels=3), y, z))
    dataset = dataset.map(lambda x, y, z: (tf.image.crop_to_bounding_box(
        x, y[0][1], y[0][0], y[1][1] - y[0][1], y[1][0] - y[0][0]), z))

    # normalisation
    dataset = dataset.map(lambda x, z:
                          (tf.cast(x, dtype=tf.float32) / 255.0, z))
    dataset = dataset.map(lambda x, z:
                          (x, tf.cast(kp2img(z), dtype=tf.float32)))
    return dataset


def kp2img(z):
    # positive channels
    indices_pos = tf.where(
        tf.reduce_sum(z, axis=1) >= 0)  # shape: [12, 1] e.g [[2], [5]...]

    indices = tf.concat([z, tf.reshape(tf.range(0, 17), (17, -1))], axis=-1)
    indices = tf.gather_nd(indices, indices_pos)

    updates = tf.ones(tf.shape(indices)[0], )
    kp = tf.scatter_nd(indices, updates, (64, 64, 17))
    return kp
