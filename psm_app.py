import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('model.sav', 'rb'))


def predict_NB(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(list(input_data.values()))

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return The prediction result from your input is: , prediction[0]


st.sidebar.header('User Input Parameters')


def main():
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

    diagnosis = ''
    if st.button('Predict!'):
        diagnosis = predict_NB(data)

    st.success(diagnosis)


if __name__ == '__main__':
    main()
