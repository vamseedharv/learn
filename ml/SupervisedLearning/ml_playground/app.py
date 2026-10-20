import streamlit as st
from importlib import import_module
import pandas as pd

st.set_page_config(page_title='Supervised ML Playground', page_icon='🧠', layout='wide')

ALGORITHMS = {
     'Regression': [
            'Linear Regression',
            'Multiple Linear Regression',
            'Ridge Regression',
            'Lasso Regression',
            'Decision Tree Regressor',
            'Random Forest Regressor',
            'KNN Regressor',
            'Gradient Boosting Regressor',
            'XGBoost Regressor',
        ],
        'Classification': [
            'Logistic Regression',
            'Decision Tree Classifier',
            'Random Forest Classifier',
            'KNN Classifier',
            'Gradient Boosting Classifier',
            'XGBoost Classifier',
        ],
}
# ALGORITHMS = {
#     'Must Master': {
#         'Regression': [
#             'Linear Regression',
#             'Multiple Linear Regression',
#             'Ridge Regression',
#             'Lasso Regression',
#             'Decision Tree Regressor',
#             'Random Forest Regressor',
#             'KNN Regressor',
#             'Gradient Boosting Regressor',
#             'XGBoost Regressor',
#         ],
#         'Classification': [
#             'Logistic Regression',
#             'Decision Tree Classifier',
#             'Random Forest Classifier',
#             'KNN Classifier',
#             'Gradient Boosting Classifier',
#             'XGBoost Classifier',
#         ],
#     },

#     'Learn and Practice Once': {
#         'Regression': [
#             'Polynomial Regression',
#             'Elastic Net',
#             'Support Vector Regression',
#             'LightGBM Regressor',
#             'CatBoost Regressor',
#         ],
#         'Classification': [
#             'Naive Bayes',
#             'Support Vector Classifier',
#             'LightGBM Classifier',
#             'CatBoost Classifier',
#         ],
#     },

#     'Conceptual Understanding': {
#         'Regression': [
#             'AdaBoost Regressor',
#             'Neural Network Regressor',
#         ],
#         'Classification': [
#             'AdaBoost Classifier',
#             'Neural Network Classifier',
#             'Linear Discriminant Analysis',
#             'Quadratic Discriminant Analysis',
#         ],
#     },
# }

st.title('🧠 Supervised ML Playground')
st.caption('One application · separate workspaces for each algorithm · implementation left to you')


with st.sidebar:
    st.header('Navigation')
    section = st.radio('Section', ['Overview', *ALGORITHMS.keys()])
    algorithm = None
    if section in ALGORITHMS:
        algorithm = st.selectbox('Algorithm', ALGORITHMS[section])
    st.divider()
    st.caption('Add your code inside algorithms/ and register it in the matching module.')

if section == 'Overview':
    st.subheader('Choose a supervised learning algorithm')
    st.info('Select Regression or Classification from the sidebar to open an algorithm workspace.')
    for category, names in ALGORITHMS.items():
        with st.expander(f'{category} ({len(names)})', expanded=True):
            st.write(' · '.join(names))
else:
    st.subheader(algorithm)
    st.caption(f'{section} → {algorithm}')
    tab_data, tab_code = st.tabs(['📁 Dataset', '🧪 Implementation'])
    with tab_data:
        st.info(f"The dataset should be located at datasets/{section}/{section}_{algorithm.lower().replace(' ', '_')}.csv")
        st.write('Dataset workspace')
        
        uploadedFile = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
        st.dataframe(uploadedFile)
        st.caption('Read and process the file in your implementation module.')
    with tab_code:
        st.write('Algorithm workspace')
        # if st.button('Run algorithm', key=f'{section}_{algorithm}_run'):
        module_name = ''.join(c.lower() if c.isalnum() else '_' for c in algorithm).strip('_')
        package = 'regression' if section == 'Regression' else 'classification'
        try:
            module = import_module(f'algorithms.{package}.{module_name}')
            module.run(section, algorithm)
        except ModuleNotFoundError as exc:
            if exc.name == f'algorithms.{package}.{module_name}':
                st.warning(f'Create algorithms/{package}/{module_name}.py with a run(uploadedFile) function.')
            else:
                st.exception(exc)
        except Exception as exc:
            st.exception(exc)
