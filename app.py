import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from src.models.train_model import dice_coef, dice_loss
from src.models.predict_model import predict
from src.visualization.visualize import visualize_prediction

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "models/2.0-model.h5"
model = load_model(
    MODEL_PATH, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef}
)

# Define upload folder and output folder
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


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
        # Save the uploaded file
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        try:
            # Make prediction
            prediction = predict(model, image_path)

            # Visualize prediction
            prediction_image = visualize_prediction(
                image_path, prediction, app.config["OUTPUT_FOLDER"]
            )

            # Remove the uploaded image after prediction
            os.remove(image_path)

            # Return the HTML with the prediction image path
            return render_template("index.html", prediction_image=prediction_image)
        except Exception as e:
            # Handle prediction error
            error_message = f"An error occurred during prediction: {str(e)}"
            return render_template("index.html", prediction_text=error_message)


if __name__ == "__main__":
    app.run(debug=True)
