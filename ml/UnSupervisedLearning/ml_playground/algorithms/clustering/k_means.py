from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

def write_explanation():
    st.write("## K-means Clustering")
    st.write(
        "K-means is an unsupervised learning algorithm that divides numeric data into K clusters. "
        "Each point belongs to one cluster, represented by its centroid (the mean of its points). "
        "It discovers groups without using outcome labels such as Pass or Fail."
    )
    st.write("### How K-means works")
    st.write("1. Choose K, the number of clusters, and prepare the numeric features. Scale features when their units or ranges differ.")
    st.write("2. Initialize K centroids. Scikit-learn uses k-means++ by default to help choose well-spaced starting centers.")
    st.write("3. Assign each point to its nearest centroid using Euclidean distance.")
    st.write("4. Update each centroid to the mean of all points assigned to that cluster.")
    st.write("5. Repeat assignment and update steps until convergence or the iteration limit is reached.")
    st.write(
        "The objective is to minimize WCSS (Within-Cluster Sum of Squares): the sum of squared "
        "distances between every point and its assigned centroid. In scikit-learn, inertia_ stores WCSS. "
        "Different starting centers can produce different solutions; n_init=10 tries ten initializations "
        "and keeps the one with the lowest inertia."
    )

    st.write("### Advantages")
    st.write("- Simple to understand and implement, with centroids that summarize each group.")
    st.write("- Usually fast and scalable for large numeric datasets when K and the number of features are moderate.")
    st.write("- Works well for compact, roughly spherical, well-separated clusters of similar size and density.")
    st.write("- Provides cluster assignments and allows new points to be assigned to the nearest learned centroid.")

    st.write("### Disadvantages")
    st.write("- K must be chosen in advance, and the data may not have a single clear best K.")
    st.write("- Sensitive to feature scales, outliers, and initialization; it can converge to a local optimum.")
    st.write("- Can perform poorly on curved or elongated clusters and clusters with unequal sizes or densities.")
    st.write("- Requires numeric features and meaningful distances; categorical data needs careful treatment.")
    st.write("- Assigns every point to a cluster, including noise, and cluster numbers have no inherent order or meaning.")

    st.write("### Ways to identify K")
    st.write(
        "1. Elbow method (WCSS): fit models for several values of K and plot their inertia. "
        "Look for a bend where additional clusters give smaller improvements. WCSS decreases as K increases, "
        "so choosing the smallest WCSS alone is not useful; an elbow may also be unclear."
    )
    st.write(
        "2. Silhouette score: compares how close points are to their own cluster versus the nearest other cluster. "
        "Scores range from -1 to 1; higher is generally better, values near 0 suggest overlap, and negative "
        "values suggest a point may fit another cluster better. Compare K values with at least two clusters "
        "and fewer clusters than samples."
    )
    st.write(
        "3. Gap statistic: compare the clustering's within-cluster dispersion with that of reference datasets "
        "with no cluster structure. Choose K using the gap statistic's selection rule, accounting for uncertainty."
    )
    st.write(
        "4. Other internal scores: compare the Calinski-Harabasz score (higher is better) and "
        "Davies-Bouldin score (lower is better) across candidate K values."
    )
    st.write(
        "5. Stability and domain knowledge: check whether similar groups recur across random seeds or "
        "resampled data, and whether their sizes and profiles are useful for the real problem. "
        "Combine these checks with the plots and scores instead of relying on one metric alone."
    )
    st.write("### About this example")
    st.write(
        "This example clusters Study_Hours and Attendance_Percent after standardizing them. "
        "Result is excluded because it is an outcome label. The elbow plot uses scaled features, "
        "while the scatter plot shows the original units."
    )
    st.write(
        "The silhouette search below reports the best K among the values tested. "
        "The final model defaults to the best silhouette K. You can select a different K after reviewing the plots."
    )
    
    


FEATURES = ["Study_Hours", "Attendance_Percent"]


def evaluate_clusters(X_scaled):
    """Compare internal clustering scores, reusing each fitted model.

    Silhouette and Calinski-Harabasz favor higher scores; Davies-Bouldin
    favors lower scores. These scores require 2 through n_samples - 1
    distinct clusters; invalid cases are reported as NaN.
    """
    max_k = min(10, len(X_scaled), len(pd.DataFrame(X_scaled).drop_duplicates()))
    models = {}
    rows = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        cluster_count = len(set(labels))
        scores = {
            "Silhouette score": float("nan"),
            "Calinski-Harabasz score": float("nan"),
            "Davies-Bouldin score": float("nan"),
        }
        if 2 <= cluster_count < len(X_scaled):
            scores.update({
                "Silhouette score": silhouette_score(X_scaled, labels),
                "Calinski-Harabasz score": calinski_harabasz_score(X_scaled, labels),
                "Davies-Bouldin score": davies_bouldin_score(X_scaled, labels),
            })
        models[k] = model
        rows.append({"K": k, "WCSS": model.inertia_, **scores})
    return pd.DataFrame(rows), models


def run(section=None, algorithm=None):
    with st.expander("Learn about K-means", expanded=False):
        write_explanation()

    st.subheader("1. Prepare the data")
    dataset_name = f"{section}_{algorithm.lower().replace(' ', '_')}.csv"
    dataset_path = Path(__file__).resolve().parents[2] / "datasets" / section / dataset_name
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
    if X.empty:
        st.error("No valid rows are available for clustering.")
        return

    st.write("Features: Study Hours and Attendance Percent. Both are standardized before clustering.")
    st.dataframe(X, use_container_width=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("2. Compare candidate K values")
    results, models = evaluate_clusters(X_scaled)
    valid_scores = results.dropna(subset=["Silhouette score"])
    best_k = (
        int(valid_scores.loc[valid_scores["Silhouette score"].idxmax(), "K"])
        if not valid_scores.empty else None
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].plot(results["K"], results["WCSS"], marker="o")
    axes[0].set(title="Elbow method", xlabel="Number of clusters (K)", ylabel="WCSS (scaled features)")
    axes[1].plot(valid_scores["K"], valid_scores["Silhouette score"], marker="o")
    axes[1].set(title="Silhouette comparison", xlabel="Number of clusters (K)", ylabel="Silhouette score")
    if best_k is not None:
        axes[1].axvline(best_k, linestyle="--", color="tab:orange", label=f"Best tested K = {best_k}")
        axes[1].legend()
    for ax, metric, direction in (
        (axes[2], "Calinski-Harabasz score", "higher is better"),
        (axes[3], "Davies-Bouldin score", "lower is better"),
    ):
        valid_metric = results.dropna(subset=[metric])
        ax.plot(valid_metric["K"], valid_metric[metric], marker="o")
        ax.set(title=f"{metric} ({direction})", xlabel="Number of clusters (K)", ylabel=metric)
    for ax in axes:
        ax.set_xticks(results["K"])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Look for an elbow in WCSS, higher silhouette and Calinski-Harabasz scores, "
        "and a lower Davies-Bouldin score. WCSS alone always favors more clusters. "
        "All scores use standardized features; the metrics may favor different K values."
    )
    st.dataframe(results, hide_index=True, use_container_width=True)
    if best_k is None:
        st.info("Silhouette scoring requires at least two distinct clusters and fewer clusters than samples. Choose K manually below.")
    else:
        st.write(f"Best K among tested values by silhouette score: **{best_k}**.")

    st.subheader("3. Inspect the selected clustering")
    options = results["K"].tolist()
    selected_k = st.selectbox(
        "Number of clusters (K)", options,
        index=options.index(best_k) if best_k is not None else 0,
        key="k_means_selected_k",
    )
    model = models[selected_k]
    selected_result = results.loc[results["K"] == selected_k].iloc[0]
    metric_names = ["WCSS", "Silhouette score", "Calinski-Harabasz score", "Davies-Bouldin score"]
    for column, metric in zip(st.columns(4), metric_names):
        score = selected_result[metric]
        column.metric(metric, f"{score:.4f}" if pd.notna(score) else "Not defined")

    centers = scaler.inverse_transform(model.cluster_centers_)
    fig, ax = plt.subplots()
    for cluster in sorted(set(model.labels_)):
        points = X.loc[model.labels_ == cluster]
        ax.scatter(points[FEATURES[0]], points[FEATURES[1]], label=f"Cluster {cluster}", alpha=0.7)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, c="black", label="Centroids")
    ax.set(title=f"K-means with K = {selected_k}", xlabel="Study Hours", ylabel="Attendance Percent")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("Cluster sizes")
    counts = pd.Series(model.labels_, name="Cluster").value_counts().sort_index()
    st.dataframe(counts.rename("Count").rename_axis("Cluster").reset_index(), hide_index=True)
    st.write("Dataset with cluster assignments")
    afterClustering = df.loc[valid_rows].assign(Cluster=model.labels_)
    st.dataframe(afterClustering, use_container_width=True)

    st.subheader(f"5. Finding mean on columns selected features {FEATURES}")
    # from cluster data based on each cluster find avg 
    # Find avg based on FEATURES
    
    afterClustering = afterClustering.groupby("Cluster")
    meanOnFeatures = afterClustering[FEATURES].mean()
    meanOnFeatures = meanOnFeatures.sort_values(FEATURES)
    st.dataframe(meanOnFeatures)

    st.subheader(f"6. Starting interpretation")
    cluster_names = {
        0: "High study, low attendance",
        1: "Low study, high attendance",
        2: "Low study, low attendance",
        3: "High study, high attendance",
    }
    # Label the cluster summary.
    meanOnFeatures["Label"] = meanOnFeatures.index.map(cluster_names)
    labeled_data = df.loc[valid_rows].assign(Cluster=model.labels_)
    labeled_data["Label"] = labeled_data["Cluster"].map(cluster_names)
    st.dataframe(labeled_data)

    # Prepare the data with same features
    # Use same scaler to transform the new data    
    # Post scaling, predict the cluster for new data and assign the label based on cluster_names

    #prepare the new data for prediction
    st.subheader(f"7. Predicting cluster for new data")
    # Generate data
    new_data_dataset_name = f"{section}_{algorithm.lower().replace(' ', '_')}_newData.csv"
    new_data_dataset_name = Path(__file__).resolve().parents[2] / "datasets" / section / new_data_dataset_name
    new_data = pd.read_csv(new_data_dataset_name)
    new_df = new_data[FEATURES]

    st.write("New data for prediction:")
    st.dataframe(new_df, use_container_width=True)

    # Use the same scaler to transform the new data
    new_data_scaled = scaler.transform(new_df)

    # Predict the cluster for new data
    new_data['Cluster'] = model.predict(new_data_scaled)
    new_data['Label'] = new_data['Cluster'].map(cluster_names)
    st.write("Predicted clusters for new data with labels:")
    st.dataframe(new_data)
