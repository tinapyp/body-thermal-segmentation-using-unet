import matplotlib.pyplot as plt
import numpy as np
from src.data.make_dataset import read_image, read_mask

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def visualize_with_mask(image_path, mask_path, prediction):
    x = read_image(image_path)
    y = read_mask(mask_path)
    y_pred = prediction > 0.5
    h, w, _ = x.shape
    white_line = np.ones((h, 10, 3))

    all_images = [x, white_line, y, white_line, mask_parse(y_pred)]
    image = np.concatenate(all_images, axis=1)

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.show()
    
def visualize_prediction(image_path, prediction):
    x = read_image(image_path)
    y_pred = prediction > 0.5
    h, w, _ = x.shape
    white_line = np.ones((h, 10, 3))

    all_images = [
        x, white_line,
        mask_parse(y_pred)
    ]
    image = np.concatenate(all_images, axis=1)

    fig = plt.figure(figsize=(12, 12))
    a = fig.add_subplot(1, 1, 1)
    plt.imshow(image)
