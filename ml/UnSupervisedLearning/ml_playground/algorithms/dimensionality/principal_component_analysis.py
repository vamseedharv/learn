import streamlit as st
from ._common import explanation, run_reducer


def write_explanation():
    explanation('pca')


def run(section=None, algorithm=None):
    with st.expander("Learn about this reduction method", expanded=False):
        write_explanation()
    run_reducer(section, algorithm, 'pca')
