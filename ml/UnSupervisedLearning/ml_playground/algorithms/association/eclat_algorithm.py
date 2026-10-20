import streamlit as st
from ._common import explanation, run_miner, eclat


def write_explanation():
    explanation('Eclat', 'Eclat stores transaction-ID sets for each item and performs depth-first intersections to count larger itemsets. Advantage: efficient vertical counting. Limitation: large transaction-ID sets and many intersections can use substantial memory.')


def run(section=None, algorithm=None):
    with st.expander("Learn about Eclat", expanded=False):
        write_explanation()
    run_miner(section, algorithm, eclat)
