from pathlib import Path
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.metrics import silhouette_score


def write_explanation():
    st.write("## Hierarchical Clustering")
    st.write(
        "Hierarchical clustering is an unsupervised learning technique that builds a hierarchy of nested "
        "clusters using similarities or distances between observations. A dendrogram (tree diagram) "
        "shows how groups join or split. Outcome labels such as Pass or Fail are not used to form the clusters."
    )
    st.write("### Types of hierarchical clustering")
    st.write("- Agglomerative (bottom-up): start with each observation in its own cluster and repeatedly merge clusters. This is the approach used by scikit-learn's AgglomerativeClustering.")
    st.write("- Divisive (top-down): start with all observations in one cluster and repeatedly split groups using a chosen splitting strategy.")

    st.write("### How agglomerative clustering works")
    st.write("1. Select meaningful features, handle missing values and outliers, and scale numeric features when their units or ranges differ.")
    st.write("2. Choose a distance metric and linkage rule. Initially, every observation is a separate cluster.")
    st.write("3. Find the pair of clusters with the smallest linkage value and merge them.")
    st.write("4. Recalculate the relationship between the new cluster and the remaining clusters using the same linkage rule.")
    st.write("5. Continue until the desired cluster count or distance threshold is reached. Building the full tree continues until one cluster remains.")
    st.write("6. Cut the tree at a chosen height to obtain a flat clustering. Different cut heights reveal different levels of detail.")

    st.write("### Distance metrics and linkage methods")
    st.write("A distance metric compares observations; a linkage rule determines which clusters merge next.")
    st.write("- Euclidean distance: straight-line distance, commonly used for scaled numeric features.")
    st.write("- Manhattan distance: sum of absolute feature differences. Cosine distance compares vector direction. Choose a metric that fits the meaning of your data and is supported by the linkage method.")
    st.write("- Single linkage: smallest distance between any pair of points in the two clusters. It can capture irregular shapes but may join groups through chains of nearby points.")
    st.write("- Complete linkage: largest distance between any pair of points in the two clusters. It favors compact groups but can be sensitive to outliers.")
    st.write("- Average linkage: average distance over all pairs of points across the two clusters. It balances the nearest-pair and farthest-pair approaches.")
    st.write("- Ward linkage: merge the pair that produces the smallest increase in total within-cluster sum of squares (WCSS). It favors compact groups and requires Euclidean distance.")
    st.write("Ward linkage is related to the WCSS objective of K-means, but its greedy merges do not generally produce the same partition as K-means.")

    st.write("### Reading a dendrogram")
    st.write("- Leaves represent observations (or aggregated groups in a truncated plot), and branches show merges.")
    st.write("- Merge height represents the linkage value. Its interpretation depends on the linkage method; it is not always a raw point-to-point distance.")
    st.write("- For a standard monotonic dendrogram, draw a horizontal cut between merge heights. The number of branches it crosses is the resulting number of clusters.")
    st.write("- A large jump in merge heights can suggest separated groups. Horizontal leaf positions are for presentation and do not represent distances.")

    st.write("### Advantages")
    st.write("- Reveals nested groups and relationships at multiple levels through a dendrogram.")
    st.write("- A full hierarchy can be built before deciding on a final K, allowing several cuts to be explored.")
    st.write("- Offers different linkage methods and compatible distance metrics for different data structures.")
    st.write("- Standard agglomerative clustering does not require randomly initialized centroids, though ties and input ordering can affect results.")

    st.write("### Disadvantages")
    st.write("- Many standard implementations require quadratic memory or work with pairwise distances, making large datasets expensive. Runtime depends on the method and implementation.")
    st.write("- Greedy merges are not undone, so an early poor merge can affect the remaining hierarchy.")
    st.write("- Results can change substantially with scaling, the distance metric, linkage, noise, and outliers.")
    st.write("- Dendrograms become difficult to read with many observations, and the best cut may be ambiguous.")
    st.write("- Standard agglomerative clustering provides no native predict method for new observations. Adding data usually requires refitting or a separate assignment strategy.")
    st.write("- Cluster IDs have no inherent ranking or meaning, and observations are assigned to groups rather than explicitly identified as noise.")

    st.write("### Ways to identify K or a cut threshold")
    st.write("1. Dendrogram cut: look for a substantial gap between successive merge heights and consider a horizontal cut within that gap. Treat this as a visual clue rather than proof of a unique best K.")
    st.write("2. Distance threshold: choose the maximum merge height appropriate to the problem. The resulting K follows from that cut; threshold values depend on scaling, the metric, and linkage.")
    st.write("3. Silhouette score: compare partitions for candidate K values using a distance metric consistent with the clustering. Values range from -1 to 1; higher is generally better, near 0 suggests overlap, and negative values suggest poor assignments. It requires between 2 and n_samples - 1 distinct labels.")
    st.write("4. WCSS or elbow method: for Euclidean numeric data, compute squared distances from each point to its cluster mean and sum them for each candidate K. Look for diminishing improvements. AgglomerativeClustering does not provide inertia_; compute WCSS separately. This measure aligns most naturally with Ward linkage.")
    st.write("5. Other internal scores: compare Calinski-Harabasz (higher is better) and Davies-Bouldin (lower is better), remembering that these centroid-based measures favor particular cluster geometries.")
    st.write("6. Stability and domain knowledge: compare cluster profiles across resampled or perturbed data and check whether group sizes and differences are meaningful. Use several forms of evidence rather than a single score.")

    st.write("### Interpreting and validating cluster labels")
    st.write("Summarize each cluster using feature means, medians, sizes, and distributions. Assign descriptive names from these profiles, with a clear reference for words such as high and low. Recheck the mapping whenever the fitted clustering changes; sorting a summary does not renumber cluster IDs.")
    st.write("A silhouette score evaluates separation, not whether a descriptive name is correct. If independent reference labels exist, adjusted Rand index or adjusted mutual information can compare partitions without assuming matching cluster IDs. A reference outcome may describe a different grouping from the one the features reveal.")

    st.write("### Connection to the K-means example")
    st.write("For a comparable experiment, use the same Study_Hours and Attendance_Percent features and the same standardization as the K-means page, keeping Result out of the clustering inputs. Compare Ward linkage with K-means, inspect a dendrogram, and evaluate several cuts using silhouette and WCSS.")
    st.write("The experiment below builds one Euclidean linkage tree, compares cuts using silhouette and WCSS, and displays cluster profiles and scatter plots in original units. Descriptive labels compare cluster means with the overall feature means.")
FEATURES = ["Study_Hours", "Attendance_Percent"]


def evaluate_cuts(X_scaled, tree):
    """Evaluate exact K cuts of the same hierarchy, including tied merge heights."""
    rows, assignments = [], {}
    for k in range(1, min(10, len(X_scaled)) + 1):
        labels = cut_tree(tree, n_clusters=[k]).ravel()
        assignments[k] = labels
        wcss = sum(
            ((points - points.mean(axis=0)) ** 2).sum()
            for cluster in np.unique(labels)
            for points in [X_scaled[labels == cluster]]
        )
        score = silhouette_score(X_scaled, labels) if 2 <= k < len(X_scaled) else np.nan
        rows.append({"K": k, "WCSS": float(wcss), "Silhouette score": score})
    return pd.DataFrame(rows), assignments


def run(section=None, algorithm=None):
    with st.expander("Learn about hierarchical clustering", expanded=False):
        write_explanation()

    st.subheader("1. Prepare the data")
    dataset_name = f"{section}_{algorithm.lower().replace(' ', '_')}.csv"
    dataset_path = Path(__file__).resolve().parents[2] / "datasets" / section / dataset_name
    if not dataset_path.is_file():
        st.error(f"Dataset not found: {dataset_path.name}")
        return
    df = pd.read_csv(dataset_path)
    missing_columns = set(FEATURES) - set(df.columns)
    if missing_columns:
        st.error(f"Missing required features: {', '.join(sorted(missing_columns))}")
        return

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid_rows = X.notna().all(axis=1) & ~X.isin([float("inf"), -float("inf")]).any(axis=1)
    omitted = int((~valid_rows).sum())
    if omitted:
        st.warning(f"Excluded {omitted} rows with missing or invalid feature values.")
    X = X.loc[valid_rows]
    if len(X) < 2:
        st.error("At least two valid rows are required to build a hierarchy.")
        return

    st.write("Features: Study Hours and Attendance Percent. Both are standardized before clustering.")
    st.dataframe(X, use_container_width=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("2. Build the hierarchy")
    method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"], key="hierarchical_linkage")
    st.write("Distance metric: Euclidean. Ward merges minimize the increase in WCSS; other methods use their respective distance rules.")
    tree = linkage(X_scaled, method=method, metric="euclidean")
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(X) > 30:
        dendrogram(tree, truncate_mode="lastp", p=30, show_leaf_counts=True, ax=ax)
        st.caption("Truncated to the last 30 groups. Parenthesized leaf counts represent merged observations.")
    else:
        dendrogram(tree, ax=ax)
    ax.set(title=f"Hierarchical dendrogram ({method} linkage)", xlabel="Observations or merged groups", ylabel="Linkage height")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("3. Compare candidate K values")
    results, assignments = evaluate_cuts(X_scaled, tree)
    valid_scores = results.dropna(subset=["Silhouette score"])
    best_k = int(valid_scores.loc[valid_scores["Silhouette score"].idxmax(), "K"]) if not valid_scores.empty else None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(results["K"], results["WCSS"], marker="o")
    axes[0].set(title="WCSS elbow", ylabel="WCSS (scaled features)")
    axes[1].plot(valid_scores["K"], valid_scores["Silhouette score"], marker="o")
    axes[1].set(title="Silhouette comparison", ylabel="Silhouette score")
    if best_k is not None:
        axes[1].axvline(best_k, color="tab:orange", linestyle="--", label=f"Best tested K = {best_k}")
        axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Number of clusters (K)")
        ax.set_xticks(results["K"])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.caption("WCSS is calculated from cluster means, not an inertia_ attribute. Compare the elbow, silhouette, and dendrogram before choosing K.")
    if best_k is None:
        st.info("No candidate has a defined silhouette score. Silhouette requires at least two clusters and fewer clusters than observations.")
    else:
        st.write(f"Best K among tested cuts by silhouette score: **{best_k}**.")

    st.subheader("4. Inspect the selected clustering")
    options = results["K"].tolist()
    selected_k = st.selectbox("Number of clusters (K)", options, index=options.index(best_k) if best_k is not None else 0, key=f"hierarchical_k_{method}")
    labels = assignments[selected_k]
    selected_result = results.loc[results["K"] == selected_k].iloc[0]
    left, right = st.columns(2)
    left.metric("WCSS", f"{selected_result['WCSS']:.2f}")
    score = selected_result["Silhouette score"]
    right.metric("Silhouette score", f"{score:.4f}" if pd.notna(score) else "Not defined")
    st.caption("K selects an exact tree cut by merge order. Tied merge heights can mean that a single horizontal distance threshold cannot produce that exact K.")

    labeled_data = df.loc[valid_rows].assign(Cluster=labels)
    fig, ax = plt.subplots()
    for cluster in np.unique(labels):
        points = X.loc[labels == cluster]
        ax.scatter(points[FEATURES[0]], points[FEATURES[1]], label=f"Cluster {cluster}", alpha=0.7)
    ax.set(title=f"{method.title()} linkage with K = {selected_k}", xlabel="Study Hours", ylabel="Attendance Percent")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("5. Summarize and interpret clusters")
    # Use the cleaned numeric features so summaries match the fitted data.
    grouped = X.assign(Cluster=labels).groupby("Cluster")
    means = grouped[FEATURES].mean()
    summary = means.copy()
    summary.insert(0, "Count", grouped.size())
    overall = X.mean()
    def describe(row):
        def level(feature):
            if np.isclose(row[feature], overall[feature]):
                return "Near-average"
            return "Above-average" if row[feature] > overall[feature] else "Below-average"
        return f"{level(FEATURES[0])} study, {level(FEATURES[1]).lower()} attendance"
    summary["Label"] = means.apply(describe, axis=1)
    labeled_data["Label"] = labeled_data["Cluster"].map(summary["Label"])
    st.write("Labels compare each cluster mean with the overall mean of the valid rows. Different clusters can share a description; these are relative profiles, not outcome predictions.")
    st.dataframe(overall.rename("Overall mean").to_frame())
    st.dataframe(summary.sort_values(FEATURES), use_container_width=True)
    st.write("Feature distributions by cluster")
    st.dataframe(grouped[FEATURES].agg(["median", "std", "min", "max"]))
    st.write("Dataset with cluster assignments and descriptive labels")
    st.dataframe(labeled_data, use_container_width=True)
