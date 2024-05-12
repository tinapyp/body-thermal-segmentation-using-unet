**Body Thermal Segmentation using U-Net**

---

**Overview:**
Body Thermal Segmentation using U-Net is a Flask-based web application that allows users to upload thermal images of human bodies and performs segmentation using a pre-trained U-Net model. The application provides a simple interface for users to upload images, makes predictions on them, and visualizes the segmentation results.

**Components:**
1. **Flask Web Application:** Handles HTTP requests and serves web pages to users.
2. **U-Net Model:** A pre-trained deep learning model implemented using TensorFlow Keras for semantic segmentation of thermal images.
3. **Image Processing Scripts:** Includes scripts for reading images, training the model, making predictions, and visualizing the results.
4. **HTML Templates:** Provides the HTML templates for the web pages rendered by the Flask application.

**File Structure:**
```
project_root/
│
├── app.py                     # Flask web application
│
├── models/
│   └── 2.0-model.h5           # Pre-trained U-Net model
│
├── data/                      # Folder for storing data
│   ├── images/                # Folder for storing thermal images
│   │   ├── image1.jpg         # Example thermal image 1
│   │   ├── image2.jpg         # Example thermal image 2
│   │   └── ...
│   └── masks/           # Folder for storing segmentation masks (if available)
│       ├── mask1.png    # Example mask 1
│       ├── mask2.png    # Example mask 2
│       └── ...
│
├── notebooks/
│   ├── data_analysis.ipynb    # Notebook for data analysis
│   ├── model_training.ipynb   # Notebook for model training
│   └── model_evaluation.ipynb # Notebook for model evaluation
│
├── references/
│   ├── journal_paper1.pdf     # Reference journal paper
│
├── src/
│   ├── data/
│   │   └── make_dataset.py    # Script for reading images
│   │
│   ├── models/
│   │   ├── train_model.py     # Script for training the model
│   │   ├── predict_model.py   # Script for making predictions
│   │   └── __init__.py
│   │
│   └── visualization/
│       └── visualize.py       # Script for visualizing predictions
│
└── templates/
│   └── index.html             # HTML template for the web interface
└── uploads/                   # temp folder for storing uploaded images
```

**Dependencies:**
- Flask
- TensorFlow
- Keras

**Usage:**
1. Ensure all dependencies are installed.
2. Place the pre-trained U-Net model (`2.0-model.h5`) in the `models/` directory.
3. Run the Flask application using `python app.py`.
4. Access the web interface via a web browser.
5. Upload a thermal image of a human body for segmentation.
6. View the segmentation prediction.

**Features:**
- Simple and intuitive web interface for uploading images.
- Real-time prediction and visualization of thermal image segmentation.
- .png image only for thermal image formats.