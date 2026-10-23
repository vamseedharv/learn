"""Interactive DBSCAN experiment using the same features as the other clustering pages."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
from ._clustering_common import separation_scores, show_score_metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURES = ["Study_Hours", "Attendance_Percent"]


def write_explanation():
    st.write("## Density-Based Clustering (DBSCAN)")
    st.write("DBSCAN groups observations in dense regions separated by sparse regions. It discovers the number of clusters from the data and can mark isolated observations as noise rather than forcing them into a group. Outcome labels such as Result are excluded from clustering.")
    st.write("### How it works")
    st.write("1. Prepare and scale numeric features, then choose eps and min_samples for a meaningful distance metric. This example uses Euclidean distance on standardized features.")
    st.write("2. Find every observation's eps neighborhood. A core point has at least min_samples observations in that neighborhood, including itself.")
    st.write("3. Connect neighboring core points and expand clusters through chains of core points.")
    st.write("4. Assign non-core points within eps of a core point as border points. Other observations are noise, labeled -1. A border point near multiple clusters can be assigned differently depending on traversal order.")
    st.write("### Parameters and ways to choose them")
    st.write("- eps is a distance radius, not the number of clusters. Too small can produce excessive noise; too large can join separate groups.")
    st.write("- min_samples is the local density requirement. Increasing it generally requires denser regions to form clusters. A starting heuristic is at least the number of features plus one, followed by inspection and tuning.")
    st.write("- k-distance plot: sort distances to each point's min_samples-th nearest observation, counting the point itself. A bend can suggest eps; it is a visual clue and may be unclear. The plotted distances must use the same scaling and metric as DBSCAN.")
    st.write("- Compare nearby parameter values using cluster sizes, noise percentage, silhouette, stability, and domain knowledge. DBSCAN does not require choosing K in advance.")
    st.write("### Advantages")
    st.write("- Can discover irregularly shaped clusters and identify noise.")
    st.write("- Does not require K or randomly initialized centers.")
    st.write("- Offers an explicit local-density interpretation for cluster membership.")
    st.write("### Disadvantages")
    st.write("- One global radius can struggle when clusters have different densities.")
    st.write("- Sensitive to scaling, distance metric, eps, and min_samples. Distances can become less informative in high dimensions.")
    st.write("- Neighborhood computations can be expensive; scikit-learn's implementation can require quadratic memory in unfavorable cases.")
    st.write("- There is no native predict method for assigning new observations. Noise indicates low density under these settings, not proof of an anomaly or a bad observation.")
    st.write("### Validation and interpretation")
    st.write("Silhouette is defined only when there are at least two clusters and fewer clusters than evaluated observations. Here it is computed on clustered observations only, excluding noise. Always inspect the retained fraction: a high score on a small retained subset can be misleading. Euclidean silhouette can also undervalue curved clusters.")
    st.write("WCSS is not DBSCAN's objective and is not used here to choose parameters. Inspect the scatter plot, noise, distributions, and cluster profiles together. Cluster IDs are arbitrary; descriptive names must follow the observed profiles.")
    st.write("### Connection to the other examples")
    st.write("The experiment uses standardized Study_Hours and Attendance_Percent, like the K-means and hierarchical pages. Plots and summaries use original units. Names compare each cluster mean with the overall valid-data mean; different clusters may share a description.")


def fit_density(X_scaled, eps, min_samples):
    """Return the fitted model and internal scores excluding noise observations."""
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    retained = model.labels_ != -1
    labels = model.labels_[retained]
    return model, separation_scores(X_scaled[retained], labels)


def run(section=None, algorithm=None):
    with st.expander("Learn about density-based clustering", expanded=False):
        write_explanation()

    st.subheader("1. Prepare the data")
    dataset_name = f"{section}_{algorithm.lower().replace(' ', '_')}.csv"
    dataset_path = Path(__file__).resolve().parents[2] / "datasets" / section / dataset_name
    if not dataset_path.is_file():
        st.error(f"Dataset not found: {dataset_path.name}")
        return
    df = pd.read_csv(dataset_path)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing required features: {', '.join(sorted(missing))}")
        return
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid_rows = np.isfinite(X).all(axis=1)
    omitted = int((~valid_rows).sum())
    if omitted:
        st.warning(f"Excluded {omitted} rows with invalid or missing feature values.")
    X = X.loc[valid_rows]
    if len(X) < 2:
        st.error("At least two valid observations are required.")
        return
    st.write("Features: Study Hours and Attendance Percent. Both are standardized before calculating distances.")
    st.dataframe(X, use_container_width=True)
    X_scaled = StandardScaler().fit_transform(X)

    st.subheader("2. Choose the density parameters")
    min_samples = int(st.number_input("min_samples (includes the observation itself)", min_value=2, max_value=len(X), value=min(5, len(X)), step=1, key="dbscan_min_samples"))
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = neighbors.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    eps = float(st.number_input("eps (radius in standardized feature space)", min_value=0.001, value=0.3, step=0.05, format="%.3f", key="dbscan_eps"))
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(X) + 1), k_distances)
    ax.axhline(eps, color="tab:orange", linestyle="--", label=f"Selected eps = {eps:.3f}")
    ax.set(title=f"Sorted {min_samples}-neighbor distances (including self)", xlabel="Observations sorted by distance", ylabel="Distance in scaled space")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Inspect the bend and try nearby radii. eps = 0.3 is an initial value, not an automatically optimized choice.")

    st.subheader("3. Inspect clusters and noise")
    model, scores = fit_density(X_scaled, eps, min_samples)
    labels = model.labels_
    cluster_ids = sorted(set(labels) - {-1})
    noise_count = int((labels == -1).sum())
    columns = st.columns(3)
    columns[0].metric("Clusters", len(cluster_ids))
    columns[1].metric("Noise observations", noise_count)
    columns[2].metric("Noise percentage", f"{100 * noise_count / len(X):.1f}%")
    show_score_metrics(scores, suffix=" (excluding noise)")
    st.caption("All separation scores exclude noise. Compare the retained fraction as parameter changes can change which observations are evaluated.")
    if not cluster_ids:
        st.warning("All observations are noise. Try a larger eps or a smaller min_samples and inspect the resulting groups.")
    elif len(cluster_ids) == 1:
        st.info("One cluster was found. Internal separation scores require at least two clusters.")
    elif pd.isna(scores["Silhouette score"]):
        st.info("Internal separation scores are undefined for this partition of the retained observations.")

    core = np.zeros(len(X), dtype=bool)
    core[model.core_sample_indices_] = True
    point_type = np.where(labels == -1, "Noise", np.where(core, "Core", "Border"))
    fig, ax = plt.subplots()
    for cluster in cluster_ids:
        mask = labels == cluster
        ax.scatter(X.loc[mask, FEATURES[0]], X.loc[mask, FEATURES[1]], label=f"Cluster {cluster}", alpha=0.7)
    if noise_count:
        mask = labels == -1
        ax.scatter(X.loc[mask, FEATURES[0]], X.loc[mask, FEATURES[1]], marker="x", color="black", label="Noise (-1)")
    ax.set(title="DBSCAN clusters", xlabel="Study Hours", ylabel="Attendance Percent")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(pd.Series(point_type).value_counts().rename_axis("Point type").reset_index(name="Count"), hide_index=True)

    st.subheader("4. Summarize and interpret clusters")
    numeric_data = X.assign(Cluster=labels)
    grouped = numeric_data[numeric_data["Cluster"] != -1].groupby("Cluster")
    means = grouped[FEATURES].mean()
    summary = means.copy()
    summary.insert(0, "Count", grouped.size())
    overall = X.mean()
    names = {}
    for cluster, row in means.iterrows():
        levels = ["Near-average" if np.isclose(row[f], overall[f]) else "Above-average" if row[f] > overall[f] else "Below-average" for f in FEATURES]
        names[cluster] = f"{levels[0]} study, {levels[1].lower()} attendance"
    summary["Label"] = pd.Series(names, dtype="object")
    st.write("Noise is excluded from cluster profiles. Relative labels use the overall mean of all valid observations, including noise.")
    st.dataframe(overall.rename("Overall mean").to_frame())
    if not summary.empty:
        st.dataframe(summary.sort_values(FEATURES), use_container_width=True)
        st.dataframe(grouped[FEATURES].agg(["median", "std", "min", "max"]))
    names[-1] = "Noise"
    labeled_data = df.loc[valid_rows].assign(Cluster=labels, Point_Type=point_type)
    labeled_data["Label"] = labeled_data["Cluster"].map(names)
    st.write("Dataset with cluster assignments, point types, and descriptive labels")
    st.dataframe(labeled_data, use_container_width=True)
