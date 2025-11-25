import os
import pickle
import pandas as pd
import streamlit as st
import random
import time

st.header("Heart Disease Prediction Using Machine Learning")

data = '''Project Objective
Heart Disease Prediction using Machine Learning
Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes.
In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.
'''

# ----------- FIXED MODEL PATH -------------
model_path = os.path.join(os.path.dirname(__file__), "ml_model", "Heart.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.subheader(data)
st.image("https://static.wixstatic.com/media/7afd4b_658deeaebf874a409dbcd95dbe47d873~mv2.gif")

# ----------- FIXED CSV PATH -------------
csv_path = os.path.join(os.path.dirname(__file__), "heart.csv")
df = pd.read_csv(csv_path)

st.sidebar.header("Select features for prediction")
st.sidebar.image("https://cloudfront-us-east-1.images.arcpublishing.com/thedailybeast/ZN65QQ2HLNNNXLMGQPUDOEB4RM.gif")

values = []
random.seed(11)

for feature in df.columns[:-1]:  # exclude target
    min_val = int(df[feature].min())
    max_val = int(df[feature].max())
    val = st.sidebar.slider(
        f"{feature}",
        min_val,
        max_val,
        random.randint(min_val, max_val)
    )
    values.append(val)

prediction_input = [values]
prediction = model.predict(prediction_input)[0]

progress = st.progress(0)
status = st.empty()
status.subheader("Predicting Heart Disease...")
loading = st.empty()
loading.image('https://media1.tenor.com/m/LLlSFiqwJGMAAAAC/beating-heart-gif.gif', width=200)

for i in range(100):
    time.sleep(0.03)
    progress.progress(i + 1)

status.empty()
loading.empty()

if prediction == 0:
    st.success("No Heart Disease Detected")
else:
    st.warning("Heart Disease Found")
