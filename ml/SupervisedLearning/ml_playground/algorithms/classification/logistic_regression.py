# Step 1: Import libraries
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def run(section=None, algorithm=None):
    # explain Logistic Regression in detail with simple examples and words
    st.write('''Logistic Regression is a statistical method used for binary classification problems, where the goal is to predict one of two possible outcomes based on input features. It models the relationship between the independent variables (features) and the dependent variable (target) using a logistic function, which outputs probabilities between 0 and 1. The predicted probabilities are then converted into class labels based on a threshold (commonly 0.5). Logistic Regression is widely used in various fields, including healthcare, finance, and marketing, due to its simplicity, interpretability, and effectiveness in handling linearly separable data.''')

    st.markdown("### Applications")
    st.markdown(
        """
    - **Medical Diagnosis:** Predicts the presence or absence of a disease based on patient symptoms and test results.
    - **Credit Scoring:** Assesses the likelihood of a borrower defaulting on a loan based on financial history and credit score.
    - **Customer Churn Prediction:** Identifies customers likely to leave a service based on usage patterns and demographics.
    - **Email Spam Detection:** Classifies emails as spam or not spam based on content and sender information.
    - **Marketing Campaign Effectiveness:** Evaluates the success of marketing campaigns by predicting customer response.
    """
    )

    # Advantages
    st.markdown("### Advantages")
    st.markdown(
        """
    - **Simplicity:** Easy to implement and understand, making it suitable for beginners.
    - **Interpretability:** Coefficients provide insights into feature importance and relationships with the target variable.
    - **Efficiency:** Computationally efficient for small to medium-sized datasets.
    - **Probabilistic Output:** Provides predicted probabilities, allowing for more nuanced decision-making.
    """
    )

    # Limitations
    st.markdown("### Limitations")
    st.markdown(
        """
    - **Linearity Assumption:** Assumes a linear relationship between features and the log-odds of the target variable.
    - **Limited to Binary Classification:** Primarily designed for binary outcomes; extensions are needed for multi-class problems.
    - **Sensitive to Outliers:** Outliers can disproportionately influence model performance and predictions.
    - **Feature Engineering Required:** May require careful feature selection and transformation for optimal performance.
    """
    )

    df = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
    st.dataframe(df)  # Display the dataset in Streamlit

    # Step 3: Separate features and target
    # Replace 'target' with your CSV's actual target column
    X = df.drop(columns=['Result'])
    y = df['Result']

    test_size = st.slider("Select test size (as a fraction of the dataset)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random State (for reproducibility)", min_value=0, value=42, step=1)
    # Step 4: Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Step 5: Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Step 6: Make predictions
    y_pred = model.predict(X_test)

    # Step 7: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Step 8: Display confusion matrix
    
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.dataframe(cm)

    # Step 9: write st inputs for features to take inputs and build a dataframe to predict the Result for columns Age,Study_Hours,Attendance_Percent,Practice_Tests,Previous_Score,Activity_Hours
    Age = st.number_input("Age", min_value=0)
    Study_Hours = st.number_input("Study Hours", min_value=0)
    Attendance_Percent = st.number_input("Attendance Percent", min_value=0.0, max_value=100.0)
    Practice_Tests = st.number_input("Practice Tests", min_value=0)
    Previous_Score = st.number_input("Previous Score", min_value=0.0, max_value=100.0)
    Activity_Hours = st.number_input("Activity Hours", min_value=0.0)

  
    probabilities = model.predict_proba(X_test)

    st.write("Prediction:", y_pred)
    st.write("Classes:", model.classes_)
    st.write("Probabilities:", probabilities)

    # Create a dataframe from the inputs
    input_data = pd.DataFrame({
        "Age": [Age],
        "Study_Hours": [Study_Hours],
        "Attendance_Percent": [Attendance_Percent],
        "Practice_Tests": [Practice_Tests],
        "Previous_Score": [Previous_Score],
        "Activity_Hours": [Activity_Hours]
    })

    # Make a prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Result: {prediction[0]}")
    

