# Supervised ML Playground

A one-page Streamlit UI skeleton for supervised learning algorithms.

## Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Select a category and algorithm from the sidebar. Each algorithm has Dataset, Implementation and Results tabs.

## Add your algorithm

Create a module in `algorithms/regression/` or `algorithms/classification/` using the algorithm name in lowercase with spaces and punctuation replaced by underscores. For example:

`algorithms/classification/logistic_regression.py`

```python
import streamlit as st

def run(uploaded_file=None):
    st.write('Your implementation goes here')
```

Click **Run algorithm** in the UI to call its `run(uploaded_file)` function. The supplied Linear Regression module is an example placeholder. Other modules can be added as you study them.

Optional packages such as scikit-learn, xgboost, lightgbm and catboost are **not** required to run the skeleton. Install them later as needed.
