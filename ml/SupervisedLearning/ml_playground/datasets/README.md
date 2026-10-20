# Machine Learning Practice Datasets

30 synthetic CSV datasets: 16 regression + 14 classification.
Each dataset has 400 rows, column headers, and a target in the LAST column.
All data is invented for learning, not real customers/students.

Files are organized in Regression/ and Classification/.
Filenames contain both task type and algorithm name.

IMPORTANT:
- You can use the SAME dataset to compare many algorithms. The filenames
  indicate suggested starting points, NOT algorithm-only restrictions.
- Regression targets are continuous numbers; classification targets are labels.
- For classification, Result is Pass/Fail, except Naive Bayes where Is_Spam is 0/1.
- Naive Bayes uses nonnegative word counts and is suited to MultinomialNB.
- CatBoost examples include text categories (Region, Plan). Encode categories
  for algorithms that require numeric inputs; CatBoost can accept cat_features.
- Polynomial Regression: create polynomial features in Python using
  sklearn.preprocessing.PolynomialFeatures. The CSV stores raw input only.
- Linear Regression, Ridge, Lasso, Elastic Net etc. can all be tested on
  suitable numeric datasets; algorithms do not require exclusive CSV files.
- Scale features where appropriate (e.g. KNN, SVM, regularized linear models,
  neural networks); fit preprocessing on training data only.
- Some models require additional packages: xgboost, lightgbm, catboost.
- No target leakage: never include the last column in X.
- Generated with random seed 20261014 for reproducibility.

## Index
Category | Algorithm | Filename | Rows | Target | Feature count
--- | --- | --- | --- | --- | ---
Regression | Linear Regression | Regression_linear_regression.csv | 400 | Exam_Score | 1
Regression | Multiple Linear Regression | Regression_multiple_linear_regression.csv | 400 | Salary | 3
Regression | Polynomial Regression | Regression_polynomial_regression.csv | 400 | Sales | 1
Regression | Ridge Regression | Regression_ridge_regression.csv | 400 | Annual_Customer_Value | 6
Regression | Lasso Regression | Regression_lasso_regression.csv | 400 | Annual_Customer_Value | 6
Regression | Elastic Net | Regression_elastic_net.csv | 400 | Annual_Customer_Value | 6
Regression | Decision Tree Regressor | Regression_decision_tree_regressor.csv | 400 | Annual_Customer_Value | 6
Regression | Random Forest Regressor | Regression_random_forest_regressor.csv | 400 | Annual_Customer_Value | 6
Regression | Support Vector Regression | Regression_support_vector_regression.csv | 400 | Annual_Customer_Value | 6
Regression | KNN Regressor | Regression_knn_regressor.csv | 400 | Annual_Customer_Value | 6
Regression | Gradient Boosting Regressor | Regression_gradient_boosting_regressor.csv | 400 | Annual_Customer_Value | 6
Regression | AdaBoost Regressor | Regression_adaboost_regressor.csv | 400 | Annual_Customer_Value | 6
Regression | XGBoost Regressor | Regression_xgboost_regressor.csv | 400 | Annual_Customer_Value | 6
Regression | LightGBM Regressor | Regression_lightgbm_regressor.csv | 400 | Annual_Customer_Value | 6
Regression | CatBoost Regressor | Regression_catboost_regressor.csv | 400 | Annual_Customer_Value | 5
Regression | Neural Network Regressor | Regression_neural_network_regressor.csv | 400 | Annual_Customer_Value | 6
Classification | Logistic Regression | Classification_logistic_regression.csv | 400 | Result | 6
Classification | KNN Classifier | Classification_knn_classifier.csv | 400 | Result | 6
Classification | Naive Bayes | Classification_naive_bayes.csv | 400 | Is_Spam | 6
Classification | Decision Tree Classifier | Classification_decision_tree_classifier.csv | 400 | Result | 6
Classification | Random Forest Classifier | Classification_random_forest_classifier.csv | 400 | Result | 6
Classification | Support Vector Classifier | Classification_support_vector_classifier.csv | 400 | Result | 6
Classification | Gradient Boosting Classifier | Classification_gradient_boosting_classifier.csv | 400 | Result | 6
Classification | AdaBoost Classifier | Classification_adaboost_classifier.csv | 400 | Result | 6
Classification | XGBoost Classifier | Classification_xgboost_classifier.csv | 400 | Result | 6
Classification | LightGBM Classifier | Classification_lightgbm_classifier.csv | 400 | Result | 6
Classification | CatBoost Classifier | Classification_catboost_classifier.csv | 400 | Result | 8
Classification | Neural Network Classifier | Classification_neural_network_classifier.csv | 400 | Result | 6
Classification | Linear Discriminant Analysis | Classification_linear_discriminant_analysis.csv | 400 | Result | 6
Classification | Quadratic Discriminant Analysis | Classification_quadratic_discriminant_analysis.csv | 400 | Result | 6
