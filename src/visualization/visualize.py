import os
import matplotlib.pyplot as plt
import numpy as np
from src.data.make_dataset import read_image, read_mask


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask.astype(np.float32)


def visualize_prediction(image_path, prediction, output_folder="static/output"):
    """
    Visualize the prediction mask and save it to a file.

    Args:
        image_path (str): Path to the input image.
        prediction (np.ndarray): Prediction mask as a 2D array.
        output_folder (str): Folder to save the visualization (default is "static/output").
    """
    # Convert prediction to a binary mask
    y_pred = prediction > 0.5

    # Parse the prediction mask into RGB
    y_pred_rgb = mask_parse(y_pred)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define the output file path
    output_path = os.path.join(
        output_folder, os.path.basename(image_path) + "_prediction.png"
    )

    # Debugging: Print the output path
    print(f"Saving prediction visualization to {output_path}")

    # Save the visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(y_pred_rgb)
    plt.axis("off")
    
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Successfully saved the image to {output_path}")
    except Exception as e:
        print(f"Error occurred while saving the image: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Image saved after retry to {output_path}")
    
    plt.close()

    return output_path