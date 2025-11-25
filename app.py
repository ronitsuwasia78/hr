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

# Path to the .pkl file inside ml_model
model_path = os.path.join(os.path.dirname(__file__), "ml_model", r"C:\Users\ronit\OneDrive\Pictures\heartdisease\Heart.pkl")

# Load the model
with open(model_path, "rb") as f:
    chatgpt = pickle.load(f)

st.subheader(data)
st.image("https://static.wixstatic.com/media/7afd4b_658deeaebf874a409dbcd95dbe47d873~mv2.gif")
# st.image('https://t-shikuro.github.io/images/heart/heart.gif')

# ✅ Load CSV from the same folder
csv_path = os.path.join(os.path.dirname(__file__), "heart.csv")
df = pd.read_csv(csv_path)

st.sidebar.header("Select feature to predict heart disease")
st.sidebar.image("https://cloudfront-us-east-1.images.arcpublishing.com/thedailybeast/ZN65QQ2HLNNNXLMGQPUDOEB4RM.gif")

all_values = []
random.seed(11)
for i in df.iloc[:, :-1]:
    min_value, max_value = df[i].agg(['min', 'max'])
    var = st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                            random.randint(int(min_value), int(max_value)))
    all_values.append(var)

final_value = [all_values]
ans = chatgpt.predict(final_value)[0]

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Heart Disease')

place = st.empty()
place.image('https://media1.tenor.com/m/LLlSFiqwJGMAAAAC/beating-heart-gif.gif', width=200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Heart Disease Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
else:
    body = 'Heart Disease Found'
    placeholder.empty()
    place.empty()
    st.warning(body)


