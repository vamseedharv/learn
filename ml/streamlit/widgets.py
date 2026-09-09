import streamlit as st
import pandas as pd
import numpy as np

st.title("Streamlit text input")
name=st.text_input("Enter your name:")

age=st.slider("Select your age:", 0, 100)

options=["Python", "Java"]

plSelected=st.selectbox("Choose your favorite language:", options)

if name:
    st.write(f"Hello, {name} ")

if age:
    st.write(f"You said your age is, {age} ")

if plSelected:
    st.write(f"And your favorite language is {plSelected}")

data = {
    "Name": ["Ram", "Krishna", "Sita", "Radha"],
    "Age": [28, 32, 27, 31],
    "City": ["Ayodhya", "Dwaraka", "Ayodhya", "Dwaraka"]
}    
df=pd.DataFrame(data)
st.write(df)

uploaded_file=st.file_uploader("Choose a CSV file", type="CSV")

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)






