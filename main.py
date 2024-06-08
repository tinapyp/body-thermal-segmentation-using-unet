import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.models import load_model
from src.models.train_model import dice_coef, dice_loss
from src.models.predict_model import predict
from src.visualization.visualize import visualize_prediction

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
        self.original_image_label = tk.Label(self.main_frame, text="Original Image")
        self.original_image_label.grid(row=1, column=0)
        self.original_image_panel = tk.Label(self.main_frame)
        self.original_image_panel.grid(row=2, column=0)

        # Processed Image Display
        self.processed_image_label = tk.Label(self.main_frame, text="Processed Image")
        self.processed_image_panel = tk.Label(self.main_frame)
        # Hide initially
        self.processed_image_label.grid_forget()
        self.processed_image_panel.grid_forget()

        # Analyze Button
        self.analyze_button = tk.Button(self.main_frame, text="Analisis", command=self.analyze_image)
        self.analyze_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Result Display
        self.result_label = tk.Label(self.main_frame, text="", font=("Arial", 14))
        self.result_label.grid(row=4, column=0, columnspan=2)

        # Load a default image
        self.image_path = "/Users/tinapyp/Dev/Freelance/FastWork/body-thermal-segmentation-using-unet/references/u-net-architecture.png"
        self.display_image(self.image_path, self.original_image_panel)

    def display_image(self, path, panel):
        img = Image.open(path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

    def analyze_image(self):
        # Load and display the original image
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.display_image(file_path, self.original_image_panel)

            try:
                # Make prediction
                prediction = predict(model, self.image_path)

                # Visualize prediction
                output_folder = "static/output/"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                prediction_image = visualize_prediction(self.image_path, prediction, output_folder)

                # Display the processed image
                self.display_image(prediction_image, self.processed_image_panel)

                # Show the processed image panel and label
                self.processed_image_label.grid(row=1, column=1)
                self.processed_image_panel.grid(row=2, column=1)

                # Display the result
                self.result_label.config(text="Luas Area Segmentasi: {}\nBagian Tubuh: {}".format(np.sum(prediction), "Tangan"))

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
