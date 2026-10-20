import streamlit as st
from ._common import explanation, run_miner, fp_growth


def write_explanation():
    explanation('FP-Growth', 'FP-Growth orders frequent items, compresses baskets into a prefix tree, and recursively mines conditional pattern bases and trees. Advantage: avoids Apriori candidate generation. Limitation: conditional trees can still grow large.')


def run(section=None, algorithm=None):
    with st.expander("Learn about FP-Growth", expanded=False):
        write_explanation()
    run_miner(section, algorithm, fp_growth)
