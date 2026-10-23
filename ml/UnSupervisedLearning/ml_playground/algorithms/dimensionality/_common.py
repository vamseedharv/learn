"""Dimensionality-reduction experiments with method-specific diagnostics."""
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import LocallyLinearEmbedding, Isomap, trustworthiness
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.sparse.csgraph import connected_components

FEATURES = ['Age', 'Study_Hours', 'Attendance_Percent', 'Practice_Tests', 'Previous_Score', 'Activity_Hours']

DETAILS = {
'pca': ('Principal Component Analysis',
'PCA is a linear projection onto orthogonal directions of greatest variance. It centers features, finds principal axes using a matrix decomposition, and orders components by explained variance. This page standardizes features first so different units do not dominate.',
'Choose components using cumulative explained variance, a scree plot, reconstruction error, and downstream validation. A target such as 90% variance is a heuristic rather than guaranteed preservation of task-relevant information.',
'Fast, deterministic with full SVD, supports transform and approximate inverse_transform, removes linear correlation, and provides explained variance and loadings.',
'Sensitive to scales and outliers. High variance need not mean useful information. Linear projections may miss curved manifolds; components may be difficult to interpret. Component signs can flip without changing the solution.'),
'nmf': ('Non-negative Matrix Factorization',
'NMF approximates a non-negative matrix X as W × H. W describes each observation using non-negative component weights; H gives non-negative feature contributions. Iterative optimization alternates updates to reduce reconstruction loss. This page uses MinMax scaling because all inputs must be non-negative.',
'Compare component counts using reconstruction error, stability across initializations, component interpretability, and held-out performance. There is no PCA-style explained-variance ratio; a lower training error alone favors more components.',
'Additive components can be easier to interpret for non-negative data such as counts or intensities. Supports transform of new observations using the learned component basis.',
'Only accepts non-negative input. Optimization is non-convex and sensitive to initialization, scaling, and component count. Components are not orthogonal or uniquely identifiable. MinMax scaling changes the additive meaning of the original values.'),
'lle': ('Locally Linear Embedding',
'LLE is a nonlinear manifold method. It finds nearby observations, estimates weights that reconstruct each point from its neighbors, and finds a low-dimensional embedding preserving those local reconstruction relationships. This page uses standard LLE on standardized features.',
'Choose neighbors and output dimensions using neighborhood preservation, stability, plots, and downstream checks. Too few neighbors can disconnect the manifold; too many can connect unrelated regions. No explained-variance ratio is available.',
'Can unfold locally linear curved structures without requiring global linear axes. Preserves local reconstruction relationships and supports approximate out-of-sample transform.',
'Sensitive to neighbors, noise, sampling density, and feature scales. Eigenproblems can be expensive or unstable. Global distances and axes are not generally interpretable, and no general inverse_transform is available.'),
'isomap': ('Isomap',
'Isomap is a nonlinear manifold method. It builds a nearest-neighbor graph, estimates geodesic distances with shortest paths, and uses classical multidimensional scaling to embed those distances. Inputs are standardized in this example.',
'Choose neighbors and component count using graph connectivity, trustworthiness, the method-specific reconstruction error, stability, and downstream checks. Too few neighbors can disconnect the graph; too many can create shortcuts across the manifold.',
'Can preserve global manifold geometry better than a linear projection when shortest-path distances approximate the true geometry. Supports approximate transform of new observations.',
'Sensitive to shortcuts, noise, uneven sampling, and disconnected graphs. Pairwise distances and eigendecomposition can be expensive. No general inverse_transform or explained-variance ratio is available; plotted axes are not original features.')
}


def explanation(kind):
    title, how, selection, advantages, disadvantages = DETAILS[kind]
    st.write(f'## {title}')
    st.write('Dimensionality reduction represents many input features using fewer coordinates. It is unsupervised: Result is excluded from fitting and may be used only for visual context.')
    st.write('### How it works')
    st.write(how)
    st.write('### Choosing the number of dimensions and parameters')
    st.write(selection)
    st.write('### Advantages')
    st.write(advantages)
    st.write('### Disadvantages')
    st.write(disadvantages)
    st.write('### Validation and interpretation')
    st.write('Trustworthiness ranges from 0 to 1 and measures whether neighbors in the embedding are valid neighbors in the input; larger is better for that local criterion. It does not assess all global structure. Reconstruction errors from different methods or preprocessing scales are not directly comparable.')
    st.write('A separated 2D picture is not proof of real classes. Check multiple parameter choices and evaluate held-out data for a predictive task. Fit preprocessing and the reducer on training data only in that setting. This page fits the full dataset for exploration.')
    st.write('### About this example')
    st.write('The six numeric student features are reduced to selected coordinates. Outcome labels do not affect fitting. PCA and NMF include feature-component tables; nonlinear embeddings have no equivalent original-feature loadings.')


def evaluate_embedding(values, embedded, n_neighbors):
    """Evaluate introduced neighbors, lost neighbors, and shared neighborhoods."""
    original = NearestNeighbors(n_neighbors=n_neighbors).fit(values).kneighbors(return_distance=False)
    reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(embedded).kneighbors(return_distance=False)
    overlap = np.mean([len(set(a) & set(b)) / n_neighbors for a, b in zip(original, reduced)])
    return {
        'Trustworthiness': trustworthiness(values, embedded, n_neighbors=n_neighbors),
        'Continuity': trustworthiness(embedded, values, n_neighbors=n_neighbors),
        'Neighbor overlap': float(overlap),
    }


def evaluate_reconstruction(values, reconstructed):
    """Feature reconstruction diagnostics in the preprocessing scale."""
    residual = values - reconstructed
    norm = np.linalg.norm(values)
    return {
        'Reconstruction MSE': float(np.mean(residual ** 2)),
        'Reconstruction RMSE': float(np.sqrt(np.mean(residual ** 2))),
        'Reconstruction MAE': float(np.mean(np.abs(residual))),
        'Relative reconstruction error': float(np.linalg.norm(residual) / norm) if norm > 0 else np.nan,
    }


def show_evaluation(metrics):
    for column, (name, value) in zip(st.columns(len(metrics)), metrics.items()):
        column.metric(name, f'{value:.4f}' if np.isfinite(value) else 'Not defined')


def run_reducer(section, algorithm, kind):
    st.subheader('1. Prepare the data')
    name=f"{section}_{algorithm.lower().replace(' ', '_')}.csv"
    path=Path(__file__).resolve().parents[2]/'datasets'/section/name
    df=pd.read_csv(path)
    missing=set(FEATURES)-set(df.columns)
    if missing:
        st.error(f'Missing features: {sorted(missing)}')
        return
    X=df[FEATURES].apply(pd.to_numeric,errors='coerce')
    valid=np.isfinite(X).all(axis=1)
    if (~valid).any(): st.warning(f'Excluded {int((~valid).sum())} invalid rows.')
    X=X.loc[valid]
    if len(X)<4 or not X.std().gt(0).any():
        st.error('At least four valid observations with some feature variation are required.')
        return
    st.dataframe(X,use_container_width=True)
    scaler=MinMaxScaler() if kind=='nmf' else StandardScaler()
    values=scaler.fit_transform(X)
    st.write('Preprocessing: '+('MinMax scaling to [0, 1].' if kind=='nmf' else 'Standardization to comparable feature scales.'))
    st.subheader('2. Choose the reduced representation')
    max_components=min(len(FEATURES),len(X)-2) if kind in ['lle','isomap'] else min(len(FEATURES),len(X))
    components=int(st.number_input('Output components',min_value=1,max_value=max_components,value=min(2,max_components),step=1,key=f'{kind}_components'))
    neighbors=None
    if kind in ['lle','isomap']:
        lower=components+1 if kind=='lle' else 2
        neighbors=int(st.number_input('Nearest neighbors',min_value=lower,max_value=len(X)-1,value=min(max(10,lower),len(X)-1),step=1,key=f'{kind}_neighbors'))
        graph=kneighbors_graph(values,n_neighbors=neighbors,include_self=False)
        count=connected_components(graph+graph.T,directed=False,return_labels=False)
        st.write(f'Neighbor graph connected components: {count}')
        if count>1:
            st.warning('The neighbor graph is disconnected. Increase neighbors or inspect separated groups before interpreting an embedding.')
            return
    if kind=='pca':
        model=PCA(n_components=components,svd_solver='full')
    elif kind=='nmf':
        model=NMF(n_components=components,init='nndsvda',random_state=42,max_iter=1000)
    elif kind=='lle':
        model=LocallyLinearEmbedding(n_components=components,n_neighbors=neighbors,method='standard',eigen_solver='arpack',random_state=42)
    else:
        model=Isomap(n_components=components,n_neighbors=neighbors,eigen_solver='arpack')
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            embedded=model.fit_transform(values)
        for message in sorted({str(w.message) for w in caught}): st.warning(message)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        st.error(f'Embedding could not be computed: {error}. Try different components or neighbors, and inspect duplicate or constant features.')
        return
    st.subheader('3. Evaluate the representation')
    evaluation_neighbors=int(st.number_input('Evaluation neighbors', min_value=1, max_value=(len(X)-1)//2, value=min(5,(len(X)-1)//2), step=1, key=f'{kind}_evaluation_neighbors'))
    show_evaluation(evaluate_embedding(values, embedded, evaluation_neighbors))
    st.caption('All three neighborhood scores range from 0 to 1; higher is better. Trustworthiness penalizes introduced neighbors, continuity penalizes lost neighbors, and overlap is the fraction of shared nearest neighbors. Scores use the selected evaluation neighborhood and preprocessed feature space; distance ties can affect them.')
    names=[f'Component {i+1}' for i in range(components)]
    if kind=='pca':
        full=PCA(svd_solver='full').fit(values)
        variance=pd.DataFrame({'Component':np.arange(1,len(full.explained_variance_ratio_)+1),'Explained variance':full.explained_variance_ratio_,'Cumulative variance':np.cumsum(full.explained_variance_ratio_)})
        st.dataframe(variance,hide_index=True)
        st.line_chart(variance.set_index('Component'))
        st.metric('Retained explained variance',f'{100*model.explained_variance_ratio_.sum():.2f}%')
        show_evaluation(evaluate_reconstruction(values, model.inverse_transform(embedded)))
        st.write('Principal-axis coefficients on standardized features (signs are arbitrary)')
        st.dataframe(pd.DataFrame(model.components_,index=names,columns=FEATURES))
    elif kind=='nmf':
        show_evaluation(evaluate_reconstruction(values, embedded @ model.components_))
        st.write('Non-negative component contributions in MinMax-scaled feature space')
        st.dataframe(pd.DataFrame(model.components_,index=names,columns=FEATURES))
    else:
        error=model.reconstruction_error_ if kind=='lle' else model.reconstruction_error()
        st.metric('Local reconstruction objective' if kind=='lle' else 'Geodesic reconstruction error',f'{error:.6f}')
        st.caption('This method-specific quantity is not feature reconstruction MSE and is not comparable to PCA or NMF reconstruction error.')
    if kind in ['pca', 'nmf']:
        st.caption('Lower reconstruction errors are better. MSE, RMSE and MAE use scaled features; relative error is the residual Frobenius norm divided by the input norm. These are training diagnostics and are not directly comparable across preprocessing scales.')
    st.subheader('4. Explore the embedding')
    reduced=pd.DataFrame(embedded,index=X.index,columns=names)
    fig,ax=plt.subplots()
    outcome=df.loc[valid,'Result'] if 'Result' in df.columns else None
    if outcome is not None:
        for label in outcome.fillna('Unknown').unique():
            mask=outcome.fillna('Unknown')==label
            ax.scatter(embedded[mask,0],embedded[mask,1] if components>1 else np.zeros(mask.sum()),label=str(label),alpha=0.7)
        ax.legend(title='Result (visual context only)')
    else:
        ax.scatter(embedded[:,0],embedded[:,1] if components>1 else np.zeros(len(X)),alpha=0.7)
    ax.set(xlabel='Component 1',ylabel='Component 2' if components>1 else 'Display baseline',title=DETAILS[kind][0])
    ax.grid(True,alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if components>2: st.caption('The plot shows only the first two components; all selected components are in the table.')
    st.dataframe(pd.concat([df.loc[valid],reduced],axis=1),use_container_width=True)
