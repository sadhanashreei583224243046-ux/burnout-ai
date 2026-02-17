import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="BurnoutAI", page_icon="🔥")

st.title("🔥 BurnoutAI - Employee Burnout Prediction")

# -----------------------------
# Create Dataset
# -----------------------------
data = {
    "WorkHours": [4, 6, 8, 10, 12, 5, 7, 9, 11, 3],
    "SleepHours": [8, 7, 6, 5, 4, 7, 6, 5, 4, 9],
    "StressLevel": [2, 4, 6, 8, 9, 3, 5, 7, 9, 1],
    "ScreenTime": [3, 5, 6, 8, 10, 4, 6, 7, 9, 2],
    "BurnoutRisk": [0, 0, 1, 2, 2, 0, 1, 1, 2, 0]
}

df = pd.DataFrame(data)

X = df[["WorkHours", "SleepHours", "StressLevel", "ScreenTime"]]
y = df["BurnoutRisk"]

# Train Model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# -----------------------------
# User Input
# -----------------------------
work = st.number_input("Work Hours per Day", min_value=0.0)
sleep = st.number_input("Sleep Hours per Day", min_value=0.0)
stress = st.slider("Stress Level (1-10)", 1, 10)
screen = st.number_input("Screen Time (hours)", min_value=0.0)

if st.button("Predict Burnout Risk"):

    prediction = model.predict([[work, sleep, stress, screen]])

    if prediction[0] == 0:
        st.success("😊 Burnout Risk: LOW")
    elif prediction[0] == 1:
        st.warning("⚠️ Burnout Risk: MEDIUM")
    else:
        st.error("🔥 Burnout Risk: HIGH — Take Care!")
 
 