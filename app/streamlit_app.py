import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/model.pkl")

st.title("ML Prediction App")

input_data = st.text_input("Enter comma-separated values")

if st.button("Predict"):
    data = np.array([float(i) for i in input_data.split(",")]).reshape(1, -1)
    prediction = model.predict(data)
    st.write("Prediction:", prediction)
