from pyexpat import model

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def run(section=None, algorithm=None):
    # TODO: Add your Linear Regression implementation here.
    st.info('Linear Regression implementation goes here.') 

    uploadedFile = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
    st.dataframe(uploadedFile)

    X = uploadedFile.iloc[:, :-1].values  # Features
    Y = uploadedFile.iloc[:, -1].values   # Target variable
    
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    
    plt.figure(figsize=(8,6)) 
    plt.scatter(X, Y, color='blue', label='Data Points') 
    plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line') 
    plt.title('Linear Regression on student exam scores Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    noOfHours = st.number_input('Enter number of hours studied to predict the score:', min_value=0.0, max_value=24.0, value=0.0, step=0.5, key='hours_input')
    if(noOfHours > 0):
        predicted_score = model.predict([[noOfHours]])
        st.success(f'Predicted Score: {predicted_score[0]:.2f}')

    st.pyplot(plt, clear_figure=True)
