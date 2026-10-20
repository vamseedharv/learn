# Step 1: Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def run(section=None, algorithm=None):
    # explain Random Forest Regression in detail with simple examples and words
    st.write('''Random Forest Regression is an ensemble learning method that combines multiple decision trees to improve predictive performance and reduce overfitting. It works by creating a "forest" of decision trees, where each tree is trained on a random subset of the data and features. The final prediction is obtained by averaging the predictions from all individual trees, which helps to smooth out the noise and capture complex relationships in the data. Random Forest Regression is robust to outliers and can handle both numerical and categorical features effectively.''')

    st.markdown("### Applications")
    st.markdown(
        """
    - **Stock Price Prediction:** Forecasts future stock prices based on historical data and market indicators.
    - **Real Estate Valuation:** Estimates property values using features like location, size, and amenities.
    - **Energy Consumption Forecasting:** Predicts energy usage patterns for efficient resource management.
    - **Healthcare Outcome Prediction:** Assesses patient outcomes based on medical history and treatment plans.
    - **Environmental Modeling:** Models climate change impacts using various environmental factors.
    """
    )

    # Advantages
    st.markdown("### Advantages")
    st.markdown(
        """
    - **High Accuracy:** Combines multiple trees to improve prediction accuracy and reduce variance.
    - **Robustness to Overfitting:** Averaging predictions from multiple trees helps prevent overfitting.
    - **Handles Missing Values:** Can handle datasets with missing values without requiring imputation.
    - **Feature Importance:** Provides insights into feature importance for better interpretability.
    """
    )

    # Limitations
    st.markdown("### Limitations")
    st.markdown(
        """
    - **Complexity:** The model can become complex and less interpretable compared to single decision trees.
    - **Computationally Intensive:** Training multiple trees can be resource-intensive, especially with large datasets.
    - **Slower Predictions:** Making predictions can be slower due to the need to aggregate results from multiple trees.
    - **Less Effective for Extrapolation:** May struggle with predicting values outside the range of the training data.
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
    st.write(f"Training set size: {len(X_train)} samples")
    st.write(f"Testing set size: {len(X_test)} samples")

    # Step 5: Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    # Step 6: Make predictions
    y_pred = model.predict(X_test)

    # Step 7: write st inputs for Age,Experience_Years,Monthly_Income,Monthly_Spending,Activity_Hours,Support_Tickets to take inputs and build a dataframe to predict the Annual_Customer_Value
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
