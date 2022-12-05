import joblib
import streamlit as st
import numpy as np

model = joblib.load('final_model')
scaler = joblib.load('scaler')

def age(model, headLength, skullWidth, totalLength, eye, chestGirth, bellyGirth):
    """Retourne la prediction de l'âge d'un opossum"""
    x = np.array([headLength, skullWidth, totalLength, eye, chestGirth, bellyGirth]).reshape(1,6)
    x_scaled = scaler.transform(x)
    return model.predict(x_scaled)[0]

st.title("Age d'un opossum")
st.subheader("Entrez les caractéritiques de l'opossum :")
headLength = st.slider('headLength (mm)',80, 110, 92)
skullWidth = st.slider('skullWidth (mm)', 40, 80, 56)
totalLength = st.slider('totalLength (cm)', 70, 100, 88)
eye = st.slider('eye distance', 10, 20, 15)
chestGirth = st.slider('chestGirth (cm)', 20, 35, 27)
bellyGirth = st.slider('bellyGirth (cm)', 20, 50, 32)

prediction = age(model, headLength, skullWidth, totalLength, eye, chestGirth, bellyGirth)
st.write("L'âge de l'opossum est de : ", round(prediction, 1), "ans")