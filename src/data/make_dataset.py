import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(image_dir, mask_dir, split=0.1):
    # Load image and mask file paths
    image_file_names = os.listdir(image_dir)
    images = [
        os.path.join(image_dir, file_name)
        for file_name in image_file_names
        if file_name.endswith(".png")
    ]

    mask_file_names = os.listdir(mask_dir)
    masks = [
        os.path.join(mask_dir, file_name)
        for file_name in mask_file_names
        if file_name.endswith(".png")
    ]

    # Sort paths to align images and masks
    images.sort()
    masks.sort()

    # Split data into train, validation, and test sets
    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(path, image_size=256):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_size, image_size))
    x = x / 255.0
    return x


def read_mask(path, image_size=256):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (image_size, image_size))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    return x
