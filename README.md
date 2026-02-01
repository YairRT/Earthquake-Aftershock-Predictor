# 🌍 Earthquake Aftershock Risk Predictor

A machine learning application that predicts the probability of aftershocks for recent earthquakes. The app uses a trained logistic regression model to analyze earthquake data and flag high-risk events that are likely to trigger aftershocks.

## Installation Instructions

## Usage

## Features

## Project Structure
The general structure of the project is as follows:

aftershock-risk/
├── data/
├── src/
│   ├── ingestion/
│   ├── features/
│   ├── training/
│   ├── inference/
│   └── monitoring/
├── api/
│   └── main.py
├── models/
├── scripts/
│   ├── train.py
│   ├── retrain.py
│   └── evaluate.py
├── Dockerfile
├── requirements.txt
└── README.md

* data/ --> Here, the raw data gotten from USGS is stored, also cleaned data and        labeled one.

* src/ingestion --> Pull information from USGS and normalization of data

* src/features --> Building of feature logic from USGS normalized data

* src/training --> Train and package a model artifact.

* src/inference --> Compute prediction from a model artifact

* src/monitoring --> Track health of data overtime and check for covariate shift and concept drift

* api/ --> The running service

* models/ --> model versions

* scripts/train --> A command you can run from terminal to train the model

* scripts/retrain --> Fetches the newest data and trains a model based on it

* scripts/evaluate --> Standard evaluation runner

## Contributing

## License

## Contact/Support