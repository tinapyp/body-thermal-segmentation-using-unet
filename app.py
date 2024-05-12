import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from src.data.make_dataset import read_image
from src.models.train_model import dice_coef, dice_loss
from src.models.predict_model import predict
from src.visualization.visualize import visualize_prediction

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "models/2.0-model"
model = load_model('models/2.0-model.h5', custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def make_prediction():
    if "file" not in request.files:
        return render_template("index.html", prediction_text="No file part")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", prediction_text="No selected file")

    if file:
        image_path = os.path.join("./uploads", file.filename)
        file.save(image_path)

        # Make prediction
        prediction = predict(model, image_path)

        # Visualize input image and prediction
        visualize_prediction(image_path, prediction)

        # Remove the uploaded image after visualization
        os.remove(image_path)

        return render_template("index.html", prediction_text="Prediction done!")


if __name__ == "__main__":
    app.run(debug=True)
