import numpy as np
import cv2
from src.data.make_dataset import read_image


def predict(model, image_path):
    x = read_image(image_path)
    y_pred = model.predict(np.expand_dims(x, axis=0))[0]
    return y_pred
