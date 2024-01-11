import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Load the trained model using pickle
model = pickle.load(open('model.sav', 'rb'))

def predict_sp(features):
    features = np.array(features)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    return prediction, probability

def main():
    st.title("Virus Prediction Model")
    st.write("Please select the features values to predict the virus species")  # Fixed the string syntax

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

    st.subheader('User Input:')
    st.write(features)  # Use the correct variable 'features' instead of 'user_input'

    if st.button("Predict Species"):
        # Convert user input features to a list
        features_list = features.values.tolist()[0]

        # Predict species and probability
        prediction, probability = predict_sp(features_list)

        # Display the results
        st.write(f"Prediction: {prediction[0]}")
        st.write(f"Probability: {probability}")

if __name__ == "__main__":
    main()
