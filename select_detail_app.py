# -*- coding: utf-8 -*-
"""Select detail app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/133q69mJKhRoKDRnTUDOpm9WVdLhB0SlB
"""

import numpy as np
import pandas as pd
import streamlit as st

st.number_input('Pick a number', 0,10)
st.text_input('Email address')
st.date_input('Travelling date')
st.time_input('School time')
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')