import streamlit as st
import pandas as pd
import numpy as np

## Title of the application
st.title("Hello Streamlit")

## display simple text
st.write("This is simple text")

## Create data frame
df = pd.DataFrame({
    'first column': [1,2,3,4,5],
    'second column': [10,11,12,13,14]
})

st.write("Here is the data frame")
st.write(df)

## Create a line chart
chart_data=pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)

