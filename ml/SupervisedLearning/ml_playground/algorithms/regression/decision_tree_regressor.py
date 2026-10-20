# Step 1: Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def run(section=None, algorithm=None):
    # explain Decision Tree Regression in detail with simple examples and words
    st.write('''Decision Tree Regression is a non-linear regression technique that uses a tree-like
    structure to model the relationship between input features and a continuous target variable. It recursively splits the data into subsets based on feature values, creating a tree where each internal node represents a decision based on a feature, and each leaf node represents a predicted value. Decision Tree Regression can capture complex relationships in the data without assuming any specific functional form, making it suitable for datasets with non-linear patterns.
    However, Decision Trees can be prone to overfitting, especially when the tree is deep and has many branches. To mitigate this, techniques such as pruning, setting a maximum depth, or using ensemble methods like Random Forests can be employed. Decision Tree Regression is widely used in various applications, including finance, healthcare, and marketing, due to its interpretability and ability to handle both numerical and categorical data.''')

    st.markdown("### Applications")
    st.markdown(
        """
    - **Customer Segmentation:** Classifies customers into distinct buyer groups based on behavior and demographics.
    - **Credit Scoring & Risk Assessment:** Determines creditworthiness and evaluates default risk in banking.
    - **Medical Diagnosis:** Assists clinicians in diagnosing conditions based on patient symptoms and test outcomes.
    - **Predictive Maintenance:** Forecasts equipment failure in industrial settings using operational metrics.
    - **Recommendation Systems:** Powers decision rules in personalized marketing and product recommendation pipelines.
    """
    )

    # Advantages
    st.markdown("### Advantages")
    st.markdown(
        """
    - **High Interpretability:** Easy to understand, visualize, and explain to non-technical stakeholders.
    - **Minimal Data Preparation:** Requires little to no data scaling, normalization, or handling of missing values.
    - **Non-Linear Relationships:** Captures complex, non-linear interactions between features effortlessly.
    - **Automated Feature Selection:** Automatically identifies and splits on the most informative features first.
    """
    )

    # Limitations
    st.markdown("### Limitations")
    st.markdown(
        """
    - **Overfitting Risk:** Prone to growing overly complex trees that memorize noise and generalize poorly.
    - **High Variance:** Small variations in the training data can result in a completely different tree structure.
    - **Bias Toward Dominant Classes:** Can struggle or become biased when dealing with heavily imbalanced datasets.
    - **Instability with Continuous Data:** Splits on continuous variables can sometimes become suboptimal or unstable.
    - **Greedy Algorithms:** Relies on local optimal splits at each node, which may miss the global optimum.
    """
    )

    df = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
    st.dataframe(df)  # Display the dataset in Streamlit

    # Step 3: Separate features and target
    # Replace 'target' with your CSV's actual target column
    X = df.drop(columns=['Annual_Customer_Value'])
    y = df['Annual_Customer_Value']

    # Step 4: Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Step 5: Model Training
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Predict
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
