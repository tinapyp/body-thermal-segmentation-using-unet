import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.models.train_model import dice_coef, dice_loss
from sklearn.metrics import confusion_matrix, classification_report
import os

# Load the pre-trained model
MODEL_PATH = "models/2.0-model.h5"
model = load_model(MODEL_PATH, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentasi Dan Analisis Suhu Terpanas Tubuh Manusia")

        # Create the main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack()

        # Title
        self.title_label = tk.Label(self.main_frame, text="Segmentasi Dan Analisis Suhu Terpanas Tubuh Manusia", font=("Arial", 18, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=1, pady=10)

        # Original Image Display
        self.original_image_label = tk.Label(self.main_frame, text="Original Image")
        self.original_image_label.grid(row=1, column=0)
        self.original_image_panel = tk.Label(self.main_frame)
        self.original_image_panel.grid(row=2, column=0)

        # Processed Image Display
        self.processed_image_label = tk.Label(self.main_frame, text="Processed Image")
        self.processed_image_panel = tk.Label(self.main_frame)

        # Initially forget the processed image display
        self.processed_image_shown = False

        # Analyze Button
        self.analyze_button = tk.Button(self.main_frame, text="Analisis", command=self.analyze_image)
        self.analyze_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Result Display
        self.result_label = tk.Label(self.main_frame, text="", font=("Arial", 14))
        self.result_label.grid(row=4, column=0, columnspan=2)

        # Load a default image
        self.image_path = "/Users/tinapyp/Dev/Freelance/FastWork/body-thermal-segmentation-using-unet/references/u-net-architecture.png"
        self.display_image(self.image_path, self.original_image_panel)

    def forget_processed_image_display(self):
        self.processed_image_label.grid_forget()
        self.processed_image_panel.grid_forget()

    def display_processed_image(self):
        self.processed_image_label.grid(row=1, column=1)
        self.processed_image_panel.grid(row=2, column=1)

    def display_image(self, path, panel):
        img = Image.open(path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        
    def analyze_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Reset the processed image panel and label
            self.processed_image_label.grid_forget()
            self.processed_image_panel.grid_forget()

            self.image_path = file_path

            try:
                # Load image
                image = cv2.imread(file_path)

                # Convert image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Thresholding to isolate the warm areas
                _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

                # Apply morphological operations to clean up the image
                kernel = np.ones((5, 5), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

                # Perform watershed segmentation
                sure_bg = cv2.dilate(opening, kernel, iterations=5)
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                markers = cv2.watershed(image, markers)
                image[markers == -1] = [0, 0, 255]  # Red edges for watershed boundaries

                # Get ground truth mask from the segmented image
                ground_truth_mask = np.zeros_like(gray)
                ground_truth_mask[markers > 1] = 255  # Set pixels inside the watershed boundaries to 255

                # Preprocess the image for UNet (resize, normalize, etc.)
                input_image = cv2.resize(image, (256, 256))
                input_image = input_image / 255.0  # Normalize the image
                input_image = np.expand_dims(input_image, axis=0)

                # Perform prediction using the UNet model
                prediction = model.predict(input_image)[0]

                # Thresholding the prediction to obtain binary mask
                threshold = 0.8  # Adjust this threshold as needed
                binary_mask = (prediction > threshold).astype(np.uint8)

                # Resize the binary mask to the original image size
                binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

                # Compute metrics
                intersection = np.logical_and(ground_truth_mask, binary_mask)
                union = np.logical_or(ground_truth_mask, binary_mask)
                iou = np.sum(intersection) / np.sum(union)

                true_positive = np.sum(np.logical_and(ground_truth_mask == 255, binary_mask == 1))
                false_positive = np.sum(np.logical_and(ground_truth_mask == 0, binary_mask == 1))
                false_negative = np.sum(np.logical_and(ground_truth_mask == 255, binary_mask == 0))

                accuracy = (true_positive + np.sum(ground_truth_mask == 0)) / np.prod(ground_truth_mask.shape)
                precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negative)
                dice_score = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)

                # Display metrics
                metrics_text = f"IoU: {iou:.4f}\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nDice Score: {dice_score:.4f}"
                self.result_label.config(text=metrics_text)

                # Create ground truth and prediction using blue and red colors
                prediction_mask = np.zeros_like(image)
                prediction_mask[:, :, 0] = binary_mask * 255  # Blue channel

                # Overlay edges on the original image
                edges_prediction = cv2.Canny(prediction_mask, 100, 200)

                overlay = image.copy()
                overlay[edges_prediction != 0] = [255, 0, 0]  # Blue for ground truth lines

                # Label the lines
                cv2.putText(overlay, "Predict", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(overlay, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Save the overlay image to a temporary path
                temp_overlay_path = "static/temp/temp_overlay.png"
                cv2.imwrite(temp_overlay_path, overlay)

                # Save the binary mask prediction image to a temporary path
                temp_prediction_path = "static/temp/temp_prediction.png"
                cv2.imwrite(temp_prediction_path, binary_mask * 255)

                # Display the overlay image in the original image panel
                self.display_image(temp_overlay_path, self.original_image_panel)

                # Display the binary mask prediction image in the processed image panel
                self.display_image(temp_prediction_path, self.processed_image_panel)

                # Show the processed image panel and label
                self.processed_image_label.grid(row=1, column=1)
                self.processed_image_panel.grid(row=2, column=1)

                # Display the processed image panel and label
                self.display_processed_image()

                self.root.update()

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")
        else:
            self.forget_processed_image_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

