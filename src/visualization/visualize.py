import os
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


def visualize_prediction(image_path, prediction, output_folder="static/output"):
    """
    Visualize the prediction and save it to a file.

    Args:
        image_path (str): Path to the input image.
        prediction (np.ndarray): Prediction mask as a 2D array.
        output_folder (str): Folder to save the visualization (default is "static/output").
    """
    # Read the original image
    x = read_image(image_path)

    # Convert prediction to a binary mask
    y_pred = prediction > 0.5

    # Parse the prediction mask into RGB
    y_pred_rgb = mask_parse(y_pred)

    # Concatenate the original image and prediction mask
    image_with_prediction = np.concatenate([x, y_pred_rgb], axis=1)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the output file path
    output_path = os.path.join(
        output_folder, os.path.basename(image_path) + "_prediction.png"
    )

    # Save the visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(image_with_prediction)
    plt.axis("off")
    try:
        plt.savefig(output_path)
    except:
        os.remove(output_path)
        plt.savefig(output_path)
    plt.close()

    return output_path
