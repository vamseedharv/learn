"""Spectral clustering experiment with an explicit affinity graph."""
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph

from ._clustering_common import prepare_data, evaluate_labels, show_results


def write_explanation():
    st.write("## Spectral Clustering")
    st.write("Spectral clustering treats observations as nodes in a similarity graph. It uses eigenvectors of a graph Laplacian to represent connectivity, then partitions that representation. It can separate curved or non-convex groups when the affinity graph captures their structure.")
    st.write("### How it works")
    st.write("1. Prepare and standardize meaningful numeric features.")
    st.write("2. Build a similarity graph using an RBF kernel or nearest-neighbor connections.")
    st.write("3. Construct a normalized graph Laplacian and compute eigenvectors to embed graph structure in a lower-dimensional space.")
    st.write("4. Partition the spectral embedding into K clusters. This example uses K-means assignment with repeated initialization and a fixed random seed.")
    st.write("### Affinity parameters")
    st.write("- RBF similarity is exp(-gamma * squared_distance). Larger gamma makes similarity more local; smaller gamma connects distant observations more strongly. gamma depends on feature scaling.")
    st.write("- Nearest-neighbor affinity connects each observation to a chosen number of nearby observations, then symmetrizes the graph. Too few neighbors may disconnect it; too many may hide useful local structure.")
    st.write("- Disconnected components can indicate separated groups or a poorly connected graph. Review component counts together with K and parameter choices; the eigenvector partition can be ambiguous when graph structure and K disagree.")
    st.write("### Choosing K")
    st.write("- Compare silhouette scores, cluster sizes, plots, and domain knowledge for candidate K values.")
    st.write("- The eigengap heuristic looks for a gap between successive smallest Laplacian eigenvalues. Near-zero eigenvalues relate to connected components. It is a diagnostic, not a guarantee of the best K; it is not calculated on this page.")
    st.write("- Check stability across affinity parameters and random seeds. A good partition should reflect useful structure rather than a single convenient score.")
    st.write("### Advantages")
    st.write("- Can separate non-convex groups that centroid-based clustering may miss.")
    st.write("- Can use graph or pairwise similarity information rather than requiring only Euclidean centroid structure.")
    st.write("### Disadvantages")
    st.write("- Requires choosing K and affinity parameters; results depend strongly on graph construction and scaling.")
    st.write("- Dense affinity matrices require quadratic storage, and eigenvector computations can be expensive. Sparse neighbor graphs help but do not remove every scaling limit.")
    st.write("- K-means assignment introduces initialization dependence. There is no native predict method for new observations, and every observation receives a cluster rather than a noise label.")
    st.write("### Validation and interpretation")
    st.write("This page reports Euclidean silhouette in standardized original feature space, not graph space. It can undervalue curved groups that spectral clustering is designed to find. WCSS is a compactness diagnostic, not spectral clustering's objective, and should not determine K alone.")
    st.write("Compare profiles and distributions in original units. IDs have no inherent ordering, and descriptive labels summarize means rather than explain the graph partition or predict Result.")
    st.write("### Connection to the other examples")
    st.write("Study_Hours and Attendance_Percent are standardized, with Result excluded, matching the other experiments. The same affinity graph is reused for all tested K values.")


def build_affinity(X_scaled, kind, parameter):
    if kind == "RBF":
        graph = rbf_kernel(X_scaled, gamma=parameter)
        np.fill_diagonal(graph, 0)
        return graph
    graph = kneighbors_graph(X_scaled, n_neighbors=int(parameter), mode="connectivity", include_self=False)
    return (graph + graph.T) * 0.5


def fit_spectral(graph, k):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=42, n_init=10).fit(graph)
    return model.labels_, sorted({str(item.message) for item in caught})


def run(section=None, algorithm=None):
    with st.expander("Learn about spectral clustering", expanded=False):
        write_explanation()
    prepared = prepare_data(section, algorithm)
    if prepared is None:
        return
    df, X, X_scaled, _ = prepared
    st.subheader("2. Build the graph and compare K values")
    kind = st.selectbox("Affinity", ["Nearest neighbors", "RBF"], key="spectral_affinity")
    if kind == "RBF":
        parameter = float(st.number_input("RBF gamma", min_value=0.01, value=1.0, step=0.1, key="spectral_gamma"))
    else:
        parameter = int(st.number_input("Neighbors per observation", min_value=1, max_value=len(X)-1, value=min(10, len(X)-1), step=1, key="spectral_neighbors"))
    graph = build_affinity(X_scaled, kind, parameter)
    components = connected_components(csr_matrix(graph), directed=False, return_labels=False)
    st.write(f"Connected components in the affinity graph: **{components}**")
    if components > 1:
        st.warning("The graph is disconnected. Inspect whether these components represent real groups; consider changing neighbors or gamma.")
    max_k = min(10, len(X)-1, len(X.drop_duplicates()))
    rows, assignments, messages = [], {}, set()
    for k in range(2, max_k+1):
        labels, notices = fit_spectral(graph, k)
        messages.update(notices)
        assignments[k] = labels
        rows.append({"K": k, **evaluate_labels(X_scaled, labels)})
    for message in sorted(messages):
        st.warning(message)
    results = pd.DataFrame(rows)
    st.dataframe(results, hide_index=True, use_container_width=True)
    valid = results.dropna(subset=["Silhouette score"])
    best_k = int(valid.loc[valid["Silhouette score"].idxmax(), "K"]) if not valid.empty else 2
    st.caption("The default K has the best tested Euclidean silhouette, when defined. Inspect graph connectivity and cluster shapes as well.")
    options = results["K"].tolist()
    selected_k = st.selectbox("Number of clusters (K)", options, index=options.index(best_k), key=f"spectral_k_{kind}")
    show_results(df, X, X_scaled, assignments[selected_k], f"Spectral clustering (K = {selected_k}, {kind})")
