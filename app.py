import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.models.train_model import dice_coef, dice_loss
from sklearn.metrics import confusion_matrix, classification_report
import os
import threading

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
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Original Image Display
        self.original_image_label = tk.Label(self.main_frame, text=" ")
        self.original_image_label.grid(row=1, column=0)
        self.original_image_panel = tk.Label(self.main_frame)
        self.original_image_panel.grid(row=2, column=0)

        # Processed Image Display
        self.processed_image_label = tk.Label(self.main_frame, text="Processed Image")
        self.processed_image_panel = tk.Label(self.main_frame)

        # Initially forget the processed image display
        self.processed_image_shown = False

        # Upload Image Button
        self.upload_image_button = tk.Button(self.main_frame, text="Upload Image", command=self.upload_image)
        self.upload_image_button.grid(row=3, column=0, padx=5)

        # Use Camera Button
        self.use_camera_button = tk.Button(self.main_frame, text="Use Camera", command=self.use_camera)
        self.use_camera_button.grid(row=3, column=1, padx=5)

        # Result Display
        self.result_label = tk.Label(self.main_frame, text="", font=("Arial", 14))
        self.result_label.grid(row=4, column=0, columnspan=2)

        # Load a default image
        # self.image_path = "references/168_as_tkaki20.png"
        # self.display_image(self.image_path, self.original_image_panel)

    def forget_processed_image_display(self):
        self.original_image_label.grid_forget()
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
        
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.analyze_image(self.image_path)
        else:
            self.forget_processed_image_display()

    def analyze_image(self, file_path):
        if not file_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return
        
        try:
            # Load image
            image = cv2.imread(file_path)

            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Thresholding to isolate the warm areas
            _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

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
            ground_truth_mask_color = np.zeros_like(image)
            ground_truth_mask_color[:, :, 2] = ground_truth_mask  # Red channel

            # Overlay prediction and ground truth on the original image
            overlay = cv2.addWeighted(image, 0.7, prediction_mask, 0.3, 0)
            overlay = cv2.addWeighted(overlay, 0.7, ground_truth_mask_color, 0.3, 0)

            # Add text indicating blue for prediction and red for ground truth
            cv2.putText(overlay, "Prediction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(overlay, "Ground Truth", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Save the overlay image to a temporary path
            temp_overlay_path = "static/temp/temp_overlay.png"
            temp_prediction_path = "static/temp/temp_prediction.png"
            cv2.imwrite(temp_overlay_path, overlay)
            cv2.imwrite(temp_prediction_path, binary_mask * 255)

            # Load the prediction mask image to display it
            self.display_image(temp_overlay_path, self.original_image_panel)
            self.display_image(temp_prediction_path, self.processed_image_panel)

            # Show the processed image panel and label
            self.processed_image_label.grid(row=1, column=1)
            self.processed_image_panel.grid(row=2, column=1)

            self.root.update()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

    def use_camera(self):
        # Check if camera is available
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not found.")
            return
        
        # Create a thread for camera capture
        camera_thread = threading.Thread(target=self.camera_capture_loop)
        camera_thread.start()

    def camera_capture_loop(self):
        while True:
            # Capture frame from camera
            ret, frame = self.cap.read()

            if ret:
                # Perform prediction on the captured frame
                self.analyze_frame(frame)

                # Check if the user closed the window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                messagebox.showerror("Error", "Failed to capture image from camera.")
                break

        # Release the camera
        self.cap.release()
        cv2.destroyAllWindows()

    def analyze_frame(self, frame):
        try:
            # Perform prediction on the frame
            # Convert BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame for processing
            resized_frame = cv2.resize(frame_rgb, (256, 256)) / 255.0

            # Perform prediction using the UNet model
            prediction = model.predict(np.expand_dims(resized_frame, axis=0))[0]

            # Thresholding the prediction to obtain binary mask
            threshold = 0.8  # Adjust this threshold as needed
            binary_mask = (prediction > threshold).astype(np.uint8)

            # Resize the binary mask to the original frame size
            binary_mask_resized = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))

            # Create an overlay of the original frame and the processed binary mask
            # overlay = cv2.addWeighted(frame_rgb, 0.7, cv2.cvtColor(binary_mask_resized * 255, cv2.COLOR_GRAY2RGB), 0.3, 0)

            # Convert the overlay to ImageTk format
            overlay_image = Image.fromarray(binary_mask_resized * 255)
            overlay_image.thumbnail((300, 300))  # Resize the overlay image to fit within the panel
            overlay_image = ImageTk.PhotoImage(overlay_image)

            # Display the processed frame in the processed image panel
            self.processed_image_panel.config(image=overlay_image)
            self.processed_image_panel.image = overlay_image

            # Convert the original frame to ImageTk format
            original_image = Image.fromarray(frame_rgb)
            original_image.thumbnail((300, 300))  # Resize the original image to fit within the panel
            original_image = ImageTk.PhotoImage(original_image)

            # Display the original frame in the original image panel
            self.original_image_panel.config(image=original_image)
            self.original_image_panel.image = original_image

            # Show the processed image panel and label
            self.processed_image_label.grid(row=1, column=1)
            self.processed_image_panel.grid(row=2, column=1)

            self.root.update()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")
    

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()