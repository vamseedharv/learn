import streamlit as st
from ._common import explanation, run_miner, apriori


def write_explanation():
    explanation('Apriori', 'Apriori starts with frequent single items, joins them into larger candidates, and prunes any candidate with an infrequent subset. Advantage: clear level-by-level pruning. Limitation: many candidates and repeated basket scans.')


def run(section=None, algorithm=None):
    with st.expander("Learn about Apriori", expanded=False):
        write_explanation()
    run_miner(section, algorithm, apriori)
