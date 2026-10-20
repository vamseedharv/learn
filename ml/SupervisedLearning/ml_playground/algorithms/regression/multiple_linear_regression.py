import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def run(section=None, algorithm=None):
    st.html('''Linear regression is a statistical method used for predictive analysis. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the data. Multiple Linear Regression specifically extends this concept to include two or more independent variables.
        Steps to perform multiple linear regression are similar to that of simple linear Regression but difference comes in the evaluation process. We can use it to find out which factor has the highest influence on the predicted output and how different variables are related to each other. Equation for multiple linear regression is:
    ''')
    if not section or not algorithm:
        st.warning('Section and algorithm are required.')
        return

    dataset_path = f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv'

    try:
        uploaded_file = pd.read_csv(dataset_path)
    except FileNotFoundError:
        st.error(f'Dataset not found: {dataset_path}')
        return

    st.dataframe(uploaded_file)

    feature_columns = ['Experience_Years', 'Training_Hours', 'Projects_Completed']
    target_column = 'Salary'

    missing_columns = [col for col in feature_columns + [target_column] if col not in uploaded_file.columns]
    if missing_columns:
        st.error(f'Dataset is missing required columns: {missing_columns}')
        return

    X = uploaded_file[feature_columns]
    y = uploaded_file[target_column]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(4, 6))
    plt.scatter(range(len(y)), y, color='blue', label='Actual Salary')
    plt.plot(range(len(y)), y_pred, color='red', linewidth=2, label='Predicted Salary')
    plt.title(f'{algorithm} on {section} dataset')
    plt.xlabel('Observation Index')
    plt.ylabel('Salary')
    plt.legend()
    plt.grid(True)
    

    st.subheader('Predict Salary')
    experience_years = st.number_input('Experience (years):', min_value=0.0, value=2.0, step=0.5)
    training_hours = st.number_input('Training hours:', min_value=0.0, value=50.0, step=5.0)
    projects_completed = st.number_input('Projects completed:', min_value=0, value=1, step=1)

    predicted_salary = model.predict([[experience_years, training_hours, projects_completed]])[0]
    st.success(f'Predicted Salary: ${predicted_salary:,.2f}')

    st.pyplot(plt, clear_figure=True)
    


