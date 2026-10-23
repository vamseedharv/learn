"""Shared preparation, evaluation, and presentation for clustering experiments."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Study_Hours", "Attendance_Percent"]


def prepare_data(section, algorithm):
    st.subheader("1. Prepare the data")
    name = f"{section}_{algorithm.lower().replace(' ', '_')}.csv"
    path = Path(__file__).resolve().parents[2] / "datasets" / section / name
    if not path.is_file():
        st.error(f"Dataset not found: {name}")
        return None
    df = pd.read_csv(path)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing required features: {', '.join(sorted(missing))}")
        return None
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(X).all(axis=1)
    if (~valid).any():
        st.warning(f"Excluded {int((~valid).sum())} rows with missing or invalid features.")
    X = X.loc[valid]
    if len(X) < 3 or len(X.drop_duplicates()) < 2:
        st.error("At least three valid observations and two distinct feature rows are required.")
        return None
    st.write("Study Hours and Attendance Percent are standardized before clustering. Result is excluded.")
    st.dataframe(X, use_container_width=True)
    scaler = StandardScaler()
    return df.loc[valid].copy(), X, scaler.fit_transform(X), scaler


def separation_scores(X_scaled, labels):
    """Internal scores for Euclidean features; undefined partitions yield NaN."""
    scores = {name: np.nan for name in (
        "Silhouette score", "Calinski-Harabasz score", "Davies-Bouldin score"
    )}
    if 2 <= len(np.unique(labels)) < len(labels):
        scores.update({
            "Silhouette score": silhouette_score(X_scaled, labels),
            "Calinski-Harabasz score": calinski_harabasz_score(X_scaled, labels),
            "Davies-Bouldin score": davies_bouldin_score(X_scaled, labels),
        })
    return scores


def show_score_metrics(scores, suffix=""):
    for column, (name, value) in zip(st.columns(len(scores)), scores.items()):
        column.metric(name + suffix, f"{value:.4f}" if pd.notna(value) else "Not defined")
    st.caption("Higher silhouette and Calinski-Harabasz scores and lower Davies-Bouldin scores are preferred. These Euclidean scores can favor compact groups and undervalue curved clusters; compare profiles and plots too.")


def show_score_comparison(results, parameter):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name, direction in zip(axes.ravel(),
            ["WCSS", "Silhouette score", "Calinski-Harabasz score", "Davies-Bouldin score"],
            ["compactness diagnostic", "higher is better", "higher is better", "lower is better"]):
        valid = results.dropna(subset=[name])
        ax.plot(valid[parameter], valid[name], marker="o")
        ax.set(title=f"{name} ({direction})", xlabel=parameter, ylabel=name)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def evaluate_labels(X_scaled, labels):
    k = len(np.unique(labels))
    wcss = sum(((X_scaled[labels == c] - X_scaled[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels))
    return {"Clusters": k, "WCSS": float(wcss), **separation_scores(X_scaled, labels)}


def show_results(df, X, X_scaled, labels, title, centers=None):
    st.subheader("3. Inspect the selected clustering")
    metrics = evaluate_labels(X_scaled, labels)
    columns = st.columns(2)
    columns[0].metric("Clusters", metrics["Clusters"])
    columns[1].metric("WCSS", f"{metrics['WCSS']:.2f}")
    score = metrics["Silhouette score"]
    show_score_metrics({name: metrics[name] for name in (
        "Silhouette score", "Calinski-Harabasz score", "Davies-Bouldin score"
    )})
    if pd.isna(score):
        st.info("Internal separation scores require at least two clusters and fewer clusters than observations.")
    fig, ax = plt.subplots()
    for cluster in np.unique(labels):
        points = X.loc[labels == cluster]
        ax.scatter(points[FEATURES[0]], points[FEATURES[1]], label=f"Cluster {cluster}", alpha=0.7)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=150, color="black", label="Mean-Shift centers")
    ax.set(title=title, xlabel="Study Hours", ylabel="Attendance Percent")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4. Summarize and interpret clusters")
    grouped = X.assign(Cluster=labels).groupby("Cluster")
    means = grouped[FEATURES].mean()
    summary = means.copy()
    summary.insert(0, "Count", grouped.size())
    overall = X.mean()
    names = {}
    for cluster, row in means.iterrows():
        levels = ["Near-average" if np.isclose(row[f], overall[f]) else "Above-average" if row[f] > overall[f] else "Below-average" for f in FEATURES]
        names[cluster] = f"{levels[0]} study, {levels[1].lower()} attendance"
    summary["Label"] = pd.Series(names)
    st.write("Descriptions compare cluster means with the overall valid-data mean. Names are relative, may repeat, and do not predict outcomes.")
    st.dataframe(overall.rename("Overall mean").to_frame())
    st.dataframe(summary.sort_values(FEATURES), use_container_width=True)
    st.dataframe(grouped[FEATURES].agg(["median", "std", "min", "max"]))
    labeled = df.assign(Cluster=labels)
    labeled["Label"] = labeled["Cluster"].map(names)
    st.write("Dataset with cluster assignments and descriptive labels")
    st.dataframe(labeled, use_container_width=True)
