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
├── app.py
├── data
│   ├── images
│       ├── image1.png
│       ├── image2.png
│       └──  image3.png
│   ├── mask
│       ├── mask1.png
│       ├── mask2.png
│       └──  mask3.png
├── docker-compose.yml
├── Dockerfile
├── models
│   └── 2.0-model.h5
├── notebooks
│   ├── 1.0-initial-unet.ipynb
│   ├── 1.1-unet.ipynb
│   ├── 2.0-unet-with-pretrained-mobile-net.ipynb
│   └── documentation.md
├── readme.md
├── references
│   ├── MobileNetV2 architecture.png
│   ├── u-net-architecture.png
│   └── U-Net: Convolutional Networks for Biomedical.pdf
├── requirements.txt
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── make_dataset.py
│   │   └── __pycache__
│   │       ├── __init__.cpython-310.pyc
│   │       ├── __init__.cpython-39.pyc
│   │       ├── make_dataset.cpython-310.pyc
│   │       └── make_dataset.cpython-39.pyc
│   ├── __init__.py
│   ├── models
│   │   ├── predict_model.py
│   │   ├── __pycache__
│   │   │   ├── predict_model.cpython-310.pyc
│   │   │   ├── predict_model.cpython-39.pyc
│   │   │   ├── train_model.cpython-310.pyc
│   │   │   └── train_model.cpython-39.pyc
│   │   └── train_model.py
│   └── visualization
│       ├── __init__.py
│       └── visualize.py
├── static
│   ├── output
│   └── uploads
└── templates
    └── index.html
```

**Dependencies:**
- Flask
- TensorFlow
- Keras

**Usage:**
1. Clone this repo
    ``` sh
    git clone https://github.com/tinapyp/body-thermal-segmentation-using-unet
    ```
2. Pull data using git lfs
   ```sh
   git lfs pull
   ```
3. Make sure have docker installed
4. Run docker
    ```sh
    docker compose up
    ```
5. Open your web browser and go to `http://127.0.0.1:5000`.

6. Use the interface to upload an image to doing prediction.

**Features:**
- Simple and intuitive web interface for uploading images.
- Real-time prediction and visualization of thermal image segmentation.
- .png image only for thermal image formats.