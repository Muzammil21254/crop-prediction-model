import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Initialize Flask app
Flask_app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Define routes
@Flask_app.route("/")
def Home():
    return render_template("index.html")  # Correct path

@Flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text=f"The predicted crop is {prediction[0]}")

# Run the app
if __name__ == "__main__":
    Flask_app.run(debug=True)


