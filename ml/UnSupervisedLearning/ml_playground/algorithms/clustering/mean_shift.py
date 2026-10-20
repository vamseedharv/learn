"""Mean-Shift clustering experiment."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import MeanShift, estimate_bandwidth

from ._clustering_common import prepare_data, evaluate_labels, show_results


def write_explanation():
    st.write("## Mean-Shift Clustering")
    st.write("Mean-Shift finds dense regions by moving candidate centers toward local averages. It discovers a cluster count from a bandwidth rather than requiring K in advance. This example uses scikit-learn's flat neighborhood kernel.")
    st.write("### How it works")
    st.write("1. Select numeric features, handle invalid values, and standardize features so distances are meaningful.")
    st.write("2. Start candidate centers at data observations. Gather observations within the bandwidth around each center.")
    st.write("3. Move each candidate to the mean of its neighbors and repeat until convergence or the iteration limit.")
    st.write("4. Merge nearby candidate centers, then assign observations to the nearest retained center. This example sets cluster_all=True, so every valid observation receives a cluster.")
    st.write("### Choosing bandwidth and identifying the cluster count")
    st.write("Bandwidth is the neighborhood radius in standardized feature space. Small values often produce many small groups; large values often merge groups. The relationship need not be strictly monotonic.")
    st.write("estimate_bandwidth provides a starting heuristic from pairwise distances using a quantile. A quantile is not K, and its estimate is not guaranteed to be optimal. The page uses a reproducible sample of up to 500 observations for the estimate.")
    st.write("Compare nearby bandwidths using the cluster count, sizes, silhouette, plots, and stability. Avoid choosing parameters from one score alone. Duplicate observations can produce a zero estimate, requiring a positive manual bandwidth.")
    st.write("### Advantages")
    st.write("- Does not require specifying K and can locate multiple density modes without spherical K-means initialization.")
    st.write("- Gives interpretable centers and can assign new observations to the nearest learned center using predict.")
    st.write("### Disadvantages")
    st.write("- Bandwidth strongly affects results; a single global bandwidth can struggle with varying densities.")
    st.write("- Iterative neighborhood searches can be slow on large datasets, and high-dimensional distances can be less informative.")
    st.write("- Sensitive to feature scales and outliers. With cluster_all=True, isolated observations are assigned rather than flagged as noise.")
    st.write("### Validation and interpretation")
    st.write("Silhouette ranges from -1 to 1 and is defined only for 2 through n_samples - 1 clusters. Higher generally indicates better Euclidean separation. WCSS is shown as a compactness diagnostic; it is not Mean-Shift's optimization objective and cannot by itself select bandwidth.")
    st.write("Interpret cluster means, spreads, and sizes in original units. IDs are arbitrary, and density centers need not equal the final assigned cluster means.")
    st.write("### Connection to the other examples")
    st.write("This page uses standardized Study_Hours and Attendance_Percent and excludes Result, matching the other clustering experiments. Compare the resulting profiles rather than expecting cluster IDs to match across algorithms.")


def run(section=None, algorithm=None):
    with st.expander("Learn about Mean-Shift", expanded=False):
        write_explanation()
    prepared = prepare_data(section, algorithm)
    if prepared is None:
        return
    df, X, X_scaled, scaler = prepared
    st.subheader("2. Choose and compare bandwidths")
    quantile = st.slider("Bandwidth estimation quantile", 0.1, 0.9, 0.3, 0.05, key="mean_shift_quantile")
    estimated = float(estimate_bandwidth(X_scaled, quantile=quantile, n_samples=min(500, len(X)), random_state=42))
    st.write(f"Estimated bandwidth: {estimated:.4f}")
    if estimated <= 0:
        st.warning("The estimate is zero. Choose a positive manual bandwidth.")
    bandwidth = float(st.number_input("Bandwidth (scaled distance)", min_value=0.001, value=max(0.001, estimated), step=0.05, format="%.3f", key=f"mean_shift_bandwidth_{quantile}"))
    factors = [0.75, 1.0, 1.25]
    rows, models = [], {}
    for factor in factors:
        candidate = max(0.001, bandwidth * factor)
        if candidate in models:
            continue
        model = MeanShift(bandwidth=candidate, cluster_all=True, max_iter=300).fit(X_scaled)
        models[candidate] = model
        rows.append({"Bandwidth": candidate, **evaluate_labels(X_scaled, model.labels_)})
    results = pd.DataFrame(rows)
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.caption("Nearby bandwidths are comparisons; the result below uses your chosen bandwidth, not an automatically selected score winner.")
    model = models[bandwidth]
    if model.n_iter_ >= 300:
        st.warning("At least one seed reached the iteration limit. Check convergence before interpreting the result.")
    show_results(df, X, X_scaled, model.labels_, f"Mean-Shift (bandwidth = {bandwidth:.3f})", scaler.inverse_transform(model.cluster_centers_))
