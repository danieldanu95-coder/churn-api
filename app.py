from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.route("/")
def home():
    return "Churn Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert input to correct order
    input_data = [data.get(feature, 0) for feature in features]

    # Convert to array
    input_array = np.array(input_data).reshape(1, -1)

    # Scale
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "churn_probability": float(probability)
    })

if __name__ == "__main__":
    import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)