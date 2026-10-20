import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def run(section=None, algorithm=None):
    st.write('''Polynomial Regression is a form of linear regression where the relationship between the independent variable (x) and the dependent variable (y) is modelled as an 
n
t
h
n 
th
  degree polynomial. It is useful when the data exhibits a non-linear relationship allowing the model to fit a curve to the data.

Need for Polynomial Regression
Non-linear Relationships: Polynomial regression is used when the relationship between the independent variable (input) and dependent variable (output) is non-linear. Unlike linear regression which fits a straight line, it fits a polynomial equation to capture the curve in the data.
Better Fit for Curved Data: When a researcher hypothesizes a curvilinear relationship, polynomial terms are added to the model. A linear model often results in residuals with noticeable patterns which shows a poor fit. It can capture these non-linear patterns effectively.
Flexibility and Complexity: It does not assume all independent variables are independent. By introducing higher-degree terms, it allows for more flexibility and can model more complex, curvilinear relationships between variables.''')

    uploadedFile = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
    st.dataframe(uploadedFile)  
    X = uploadedFile.iloc[:, :-1].values  # Features
    y = uploadedFile.iloc[:, -1].values   # Target variable
    lin = LinearRegression()

    lin.fit(X, y)
    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(X)

    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)

    plt.scatter(X, y, color='blue')

    plt.plot(X, lin2.predict(X_poly), color='red')
    plt.title('Polynomial Regression')
    plt.xlabel('Temperature')
    plt.ylabel('Pressure')

    plt.show()
    noOfHours = st.number_input('Enter budget:', min_value=0.0, max_value=240000000000.0, value=0.0, step=0.5, key='hours_input')
    if(noOfHours > 0):
        predict_poly = poly.fit_transform([[noOfHours]])
        predicted_score = lin2.predict(predict_poly)
        st.success(f'Predicted Score: {predicted_score[0]:.2f}')
    st.pyplot(plt, clear_figure=True)