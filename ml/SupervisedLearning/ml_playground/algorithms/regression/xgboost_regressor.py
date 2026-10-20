# Step 1: Import libraries
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def run(section=None, algorithm=None):
    # explain XGBoost Regression in detail with simple examples and words
    st.write('''XGBoost Regression is an ensemble learning technique that builds a predictive model by combining multiple weak learners, typically decision trees, in a sequential manner. Each subsequent tree is trained to correct the errors made by the previous trees, focusing on the residuals of the predictions. This iterative process allows XGBoost to create a strong predictive model that can capture complex relationships in the data. It is particularly effective for regression tasks where the target variable is continuous and can handle various types of data, including numerical and categorical features.''')

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
    - **High Accuracy:** Combines multiple weak learners to improve prediction accuracy and reduce variance.
    - **Handles Complex Relationships:** Can capture non-linear interactions between features effectively.
    - **Robustness to Outliers:** Less sensitive to outliers compared to some other regression techniques.
    - **Feature Importance:** Provides insights into feature importance for better interpretability.
    """
    )

    # Limitations
    st.markdown("### Limitations")
    st.markdown(
        """
    - **Computationally Intensive:** Training multiple trees can be resource-intensive, especially with large datasets.
    - **Slower Predictions:** Making predictions can be slower due to the need to aggregate results from multiple trees.
    - **Overfitting Risk:** Can overfit if not properly tuned, especially with deep trees or too many iterations.
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
    SEED = random_state
    # Step 4: Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Step 5: Model Training
    learning_rate = st.slider("Select learning rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    n_estimators = st.slider("Select number of estimators", min_value=10, max_value=1000, value=300, step=10)
    max_depth = st.slider("Select maximum depth", min_value=1, max_value=10, value=1, step=1)
    # below two reduces overfitting
    reglambda = st.slider("Select regularization parameter (lambda)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    regalpha = st.slider("Select regularization parameter (alpha)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    gbr = XGBRegressor(loss='absolute_error',
                                learning_rate=learning_rate,
                                n_estimators=n_estimators,
                                max_depth = max_depth, 
                                random_state = SEED,
                                max_features = 5,
                                reg_lambda = reglambda,
                                reg_alpha = regalpha
                                )

    gbr.fit(X_train, y_train)

    # Step 6: Make predictions on the test data
    y_pred = gbr.predict(X_test)

    # Step 7: Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R² Score: {r2}")

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
        prediction = gbr.predict(input_data)
        st.write(f"Predicted Annual Customer Value: ${prediction[0]:,.2f}")
    