import streamlit as st
from ._common import explanation, run_miner, fp_growth


def write_explanation():
    explanation('Efficient Tree-based Algorithms (FP-tree example)', 'This menu entry is a family rather than a single defined algorithm. This page demonstrates FP-tree compression and conditional-tree mining by reusing FP-Growth. It is explicitly the same miner as the FP-Growth page, not a separate algorithm or a performance benchmark.')


def run(section=None, algorithm=None):
    with st.expander("Learn about Efficient Tree-based Algorithms (FP-tree example)", expanded=False):
        write_explanation()
    run_miner(section, algorithm, fp_growth)
