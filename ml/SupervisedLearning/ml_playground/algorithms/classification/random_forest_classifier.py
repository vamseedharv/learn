from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def run(section=None, algorithm=None):
    # explain Random Forest Classification in detail with simple examples and words
    st.write('''Random Forest Classification is an ensemble learning method that combines multiple decision trees to improve classification accuracy and reduce overfitting. Each decision tree in the forest is trained on a random subset of the training data and features, allowing the model to capture diverse patterns and relationships in the data. The final prediction is made by aggregating the predictions from all individual trees, typically through majority voting for classification tasks. Random Forest Classification is widely used in various applications, including finance, healthcare, and marketing, due to its robustness, interpretability, and ability to handle high-dimensional datasets.''')


    df = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
    st.dataframe(df)  # Display the dataset in Streamlit

    # Step 3: Separate features and target

    # Replace 'target' with your CSV's actual target column
    X = df.drop(columns=['Result'])
    y = df['Result']

    # Step 4: Split training and testing data
    test_size = st.slider("Select test size (as a fraction of the dataset)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random State (for reproducibility)", min_value=0, value=42, step=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Step 5: Train the Random Forest Classification model
    model = RandomForestClassifier(
        random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Step 7: Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.dataframe(cm)

    # Step 8: Write Streamlit inputs for features to take inputs and build a dataframe to predict the Result for columns Age, Study_Hours, Attendance_Percent, Practice_Tests, Previous_Score, Activity_Hours
    Age = st.number_input("Age", min_value=0)
    Study_Hours = st.number_input("Study Hours", min_value=0)
    Attendance_Percent = st.number_input("Attendance Percent", min_value=0, max_value=100)
    Practice_Tests = st.number_input("Practice Tests", min_value=0)
    Previous_Score = st.number_input("Previous Score", min_value=0, max_value=100)
    Activity_Hours = st.number_input("Activity Hours", min_value=0)

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
