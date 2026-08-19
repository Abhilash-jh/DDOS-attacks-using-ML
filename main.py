import os
import subprocess
import webbrowser
import time

required_files = [
    "saved_models/Ensemble.joblib",
    "saved_models/scaler.joblib",
    "saved_models/label_encoder.joblib",
    "saved_models/selected_features.joblib"
]

missing = [f for f in required_files if not os.path.exists(f)]

print("=" * 60)
print("ML Based DDoS Detection System")
print("=" * 60)

# Check whether model already exists
if missing:

    print("Some required files are missing:")
for f in missing:
    print(" -", f)

    print("\nTraining models...\n")
    print("Training Machine Learning Models...\n")

    subprocess.run(["python", "ddos_ml_classifier.py"])

else:

    print("\nExisting trained model found.")
    print("Skipping training...\n")

print("Starting Dashboard...")

subprocess.Popen(["python", "app.py"])

time.sleep(3)

webbrowser.open("http://127.0.0.1:5000")

print("\nDashboard Started Successfully.")