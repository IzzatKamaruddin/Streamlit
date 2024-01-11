# model.py
import pickle
import numpy as np

def load_model(file_path):
    return pickle.load(open(file_path, 'rb'))

def predict_species(model, features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction, probability
