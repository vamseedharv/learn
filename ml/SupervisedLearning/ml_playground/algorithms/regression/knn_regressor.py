# Step 1: Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run(section=None, algorithm=None):
    # explain KNN Regression in detail with simple examples and words
    st.write('''K-Nearest Neighbors (KNN) Regression is a non-parametric regression technique that predicts the value of a continuous target variable based on the values of its k-nearest neighbors in the feature space. The algorithm identifies the k closest data points to a given input point and calculates the average of their target values to make a prediction. KNN Regression is simple to understand and implement, making it suitable for small datasets with low-dimensional features. However, it can be computationally expensive for large datasets and may struggle with high-dimensional data due to the curse of dimensionality.''')

    st.markdown("### Applications")
    st.markdown(
        """
    - **House Price Prediction:** Estimates property values based on features like location, size, and amenities.
    - **Stock Market Forecasting:** Predicts future stock prices using historical data and market indicators.
    - **Medical Diagnosis:** Assists in predicting patient outcomes based on similar cases and medical history.
    - **Energy Consumption Forecasting:** Predicts energy usage patterns for efficient resource management.
    - **Environmental Modeling:** Models climate change impacts using various environmental factors.
    """
    )

    # Advantages
    st.markdown("### Advantages")
    st.markdown(
        """
    - **Simplicity:** Easy to understand and implement, making it accessible for beginners.
    - **Non-Parametric:** Does not assume any specific functional form, allowing it to capture complex relationships.
    - **Adaptability:** Can be used for both regression and classification tasks with minimal modifications.
    - **No Training Phase:** KNN does not require an explicit training phase, as predictions are made directly from the training data.
    """
    )

    # Limitations
    st.markdown("### Limitations")
    st.markdown(
        """
    - **Computationally Expensive:** Can be slow for large datasets due to the need to compute distances for all points.
    - **Curse of Dimensionality:** Performance can degrade in high-dimensional spaces where distances become less meaningful.
    - **Sensitive to Noise:** Outliers or noisy data can significantly affect predictions, leading to poor performance.
    - **Choice of k:** Selecting an appropriate value for k is crucial, as too small or too large values can lead to underfitting or overfitting.
    """
    )

    df = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
    st.dataframe(df)  # Display the dataset in Streamlit

    # Step 3: Separate features and target
    # Replace 'target' with your CSV's actual target column
    X = df.drop(columns=['Annual_Customer_Value'])
    y = df['Annual_Customer_Value']

    test_size = st.slider("Select test size (as a fraction of the dataset)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random State (for reproducibility)", min_value=0, value=42, step=1)

    # Step 4: Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Step 5: Model Training
    n_neighbors = st.slider("Select number of neighbors (k)", min_value=1, max_value=20, value=5, step=1)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)

    # Step 6: Fit the model to the training data
    model.fit(X_train, y_train)

    # Step 7: Make predictions on the test data
    y_pred = model.predict(X_test)

    # Step 8: Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² Score: {r2}")

    # Step 9: Write Streamlit inputs for Age, Experience_Years, Monthly_Income, Monthly_Spending, Activity_Hours, Support_Tickets to take inputs and build a dataframe to predict the Annual_Customer_Value
    st.write("### Predict Annual Customer Value")
    age = st.number_input("Age", min_value=0)
    experience_years = st.number_input("Experience (Years)", min_value=0)
    monthly_income = st.number_input("Monthly Income", min_value=0.0)
    monthly_spending = st.number_input("Monthly Spending", min_value=0.0)
    activity_hours = st.number_input("Activity Hours", min_value=0.0)
    support_tickets = st.number_input("Support Tickets", min_value=0)
    # Create a dataframe from the inputs
    input_data = pd.DataFrame({
        "Age": [age],
        "Experience_Years": [experience_years],
        "Monthly_Income": [monthly_income],
        "Monthly_Spending": [monthly_spending],
        "Activity_Hours": [activity_hours],
        "Support_Tickets": [support_tickets]
    })
    # Make a prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write(f"Predicted Annual Customer Value: ${prediction[0]:,.2f}")