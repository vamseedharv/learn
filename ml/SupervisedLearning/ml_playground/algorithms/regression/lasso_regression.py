# Step 1: Import libraries
import streamlit as st
import pandas as pd
import numpy as np
# required imports for lasso regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def run(section=None, algorithm=None):
    # Explain Lasso Regression in detail with simple examples and words
    st.write('''Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a type of linear regression that includes a regularization term in the cost function. This regularization term helps to prevent overfitting by penalizing large coefficients in the model. The regularization term is controlled by a hyperparameter called alpha (α). A higher value of alpha increases the amount of regularization, leading to smaller coefficients and a simpler model, while a lower value of alpha allows for more complex models with larger coefficients.
''')
    df = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
    st.dataframe(df)  # Display the dataset in Streamlit

    # Step 3: Separate features and target
    # Replace 'target' with your CSV's actual target column
    X = df.drop(columns=["Annual_Customer_Value"])
    y = df["Annual_Customer_Value"]

    # Step 4: Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Step 5: Feature scaling
  

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 6: Train Lasso Regression
    model = Lasso(alpha=1)
    model.fit(X_train_scaled, y_train)

    # Step 7: Predict
    y_pred = model.predict(X_test_scaled)

    # Step 8: Evaluate
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    st.write("R2 Score:", r2_score(y_test, y_pred))
    # explain the evaluation metrics
    st.write("MSE (Mean Squared Error): The average of the squared differences between the predicted and actual values. Lower values indicate better model performance.")
    st.write("RMSE (Root Mean Squared Error): The square root of the MSE. It is in the same units as the target variable and provides an interpretable measure of the model's accuracy.")
    st.write("R2 Score (Coefficient of Determination): The proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating better model performance.")

    # Step 9: Check coefficients
    st.write("Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)
    st.write("Feature Names:", X.columns.tolist())  # Display the feature names
    st.write("Alpha:", model.alpha)  # Display the alpha value used in Lasso Regression

    newDataRaw = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}_predict_data.csv')
    newData = newDataRaw.iloc[0].values  # Assuming you want to predict for the first row
    st.write("New Data for Prediction:", newData)
    newDataTransformed = scaler.transform([list(newData)])
    prediction = model.predict(newDataTransformed)
    st.write("Predicted Annual Customer Value for new data:", prediction[0])