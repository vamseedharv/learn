# Step 1: Import libraries
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

def run(section=None, algorithm=None):
    st.write('''Decision Tree Classification is a supervised learning algorithm used for classification tasks. It builds a tree-like model of decisions based on input features to predict the class label of an instance. The algorithm recursively splits the dataset into subsets based on the feature that provides the best separation of classes, creating a tree structure where each internal node represents a decision based on a feature, and each leaf node represents a predicted class label. Decision Tree Classification is easy to interpret and visualize, making it popular in various applications such as customer segmentation, credit scoring, and medical diagnosis.''')    

    st.markdown("### Applications")
    st.markdown(
        """
    - **Customer Segmentation:** Classifies customers into distinct buyer groups based on behavior and demographics.
    - **Credit Scoring & Risk Assessment:** Determines creditworthiness and evaluates default risk in banking
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

    # Step 5: Model Training
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Step 6: Predict
    y_pred = model.predict(X_test)

    # Step 7: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}") 

    # Step 8: Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    st.write("Confusion Matrix:")
    st.dataframe(cm)

    # Step 9: write st inputs for features to take inputs and build a dataframe to predict the target for columns Age,Study_Hours,Attendance_Percent,Practice_Tests,Previous_Score,Activity_Hours   
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

