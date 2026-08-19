from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_DIR = "saved_models"

# Load model and preprocessing objects
model = joblib.load(os.path.join(MODEL_DIR, "Ensemble.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))


@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = [
        data["Total_Fwd_Packets"],
        data["Total_Bwd_Packets"],
        data["Total_Length_Fwd"],
        data["Total_Length_Bwd"],
        data["Fwd_Packet_Len_Max"],
        data["Fwd_Packet_Len_Std"],
        data["Bwd_Packet_Len_Min"],
        data["Flow_Bytes_per_s"],
        data["Flow_IAT_Max"],
        data["Flow_IAT_Min"],
        data["Fwd_IAT_Total"],
        data["Fwd_IAT_Mean"],
        data["Fwd_IAT_Max"],
        data["Bwd_IAT_Total"],
        data["Bwd_IAT_Std"],
        data["Bwd_PSH_Flags"],
        data["Fwd_URG_Flags"],
        data["Fwd_Header_Len"],
        data["Bwd_Header_Len"],
        data["Fwd_Packets_per_s"]
    ]

    X = np.array(features).reshape(1, -1)

    X = scaler.transform(X)

    prediction = model.predict(X)[0]

    confidence = float(np.max(model.predict_proba(X)) * 100)

    prediction = label_encoder.inverse_transform([prediction])[0]

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)