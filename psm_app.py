import numpy as np
import pandas as pd
import streamlit as st
from model import load_model, predict_species

model = load_model('model.sav')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Nucleus = st.sidebar.slider('Nucleus', 0.0, 1.0, 0.5)
    Exosome = st.sidebar.slider('Exosome', 0.0, 1.0, 0.5)
    Ribosome = st.sidebar.slider('Ribosome', 0.0, 1.0, 0.5)
    Membrane = st.sidebar.slider('Membrane', 0.0, 1.0, 0.5)
    Endoplasmic_Reticulum = st.sidebar.slider('Endoplasmic Reticulum', 0.0, 1.0, 0.5)
    Cytosol = st.sidebar.slider('Cytosol', 0.0, 1.0, 0.5)

    data = {
        'Nucleus': Nucleus,
        'Exosome': Exosome,
        'Ribosome': Ribosome,
        'Membrane': Membrane,
        'Endoplasmic_Reticulum': Endoplasmic_Reticulum,
        'Cytosol': Cytosol
    }

    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

st.subheader('User Input:')
st.write(user_input)
