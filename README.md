# ML-Based DDoS Attack Detection and Prediction

## Overview

This project is a Machine Learning-based web application for detecting DDoS attacks from network flow features. It uses a pre-trained Ensemble Machine Learning model and provides predictions through a Flask-based dashboard.

## Features

* Machine Learning based DDoS Detection
* Ensemble Voting Classifier
* Flask Web Dashboard
* Manual Flow Prediction
* Performance Metrics
* Model Architecture Visualization
* Confidence Score

## Project Structure

```
Project/
│
├── app.py
├── main.py
├── ddos_ml_classifier.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── dashboard.html
│
├── saved_models/
│   ├── *.joblib
│
└── results/
    ├── *.png
    └── *.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

If the trained model files are missing, `main.py` will automatically run `ddos_ml_classifier.py` to generate the required model files before starting the Flask application.

## Technologies Used

* Python
* Flask
* Scikit-learn
* NumPy
* Pandas
* Joblib
* HTML
* CSS
* JavaScript

