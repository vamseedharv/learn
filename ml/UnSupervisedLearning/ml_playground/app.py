import streamlit as st
from importlib import import_module
import pandas as pd

st.set_page_config(page_title='UnSupervised ML Playground', page_icon='🧠', layout='wide')



ALGORITHMS = {
    'Clustering': [
            'K-means',
            'Hierarchical',
            'Density-Based',
            'Mean-Shift',
            'Spectral',
        ],
        'Association': [
            'Apriori Algorithm',
            'FP-Growth Algorithm',
            'Eclat Algorithm',
            'Efficient Tree-based Algorithms',
        ],
        'Dimensionality': [
            'Principal Component Analysis',
            'Non-negative Matrix Factorization',
            'Locally Linear Embedding',
            'Isomap',
        ],
}

with st.sidebar:
    st.header('Navigation')
    section = st.radio('Section', ['Overview', *ALGORITHMS.keys()])
    algorithm = None
    if section in ALGORITHMS:
        algorithm = st.selectbox('Algorithm', ALGORITHMS[section])
    st.divider()
    st.caption('Add your code inside algorithms/ and register it in the matching module.')

st.title('🧠 UnSupervised ML Playground')
st.caption('One application · separate workspaces for each algorithm · implementation left to you')

st.write("Unsupervised Learning is a type of Machine Learning where the model learns patterns, similarities, groups, or structures from data without a target variable (label).")

st.write("In Supervised Learning, you tell the model what to predict. In Unsupervised Learning, you give the model data and ask it to discover meaningful patterns.")

# Write st.write in detail about unsupervised learning
with st.expander(f'{section}'):
    if(section == "Clustering"):
        st.markdown("### Clustering")
        st.markdown("Clustering is an unsupervised machine learning technique that groups unlabeled data into clusters based on similarity. Its goal is to discover patterns or relationships within the data without any prior knowledge of categories or labels.")
        st.markdown(
            """
        - Groups data points that share similar features or characteristics.
        - Helps find natural groupings in raw, unclassified data.
        - Commonly used for customer segmentation, anomaly detection and data organization.
        - Works purely from the input data without any output labels.
        - Enables understanding of data structure for further analysis or decision-making.    
        """
        )
        st.markdown("## Some common clustering algorithms:")
        st.markdown(
            """
        - K-means Clustering: Groups data into K clusters based on how close the points are to each other.
        - Hierarchical Clustering: Creates clusters by building a tree step-by-step, either merging or splitting groups.
        - Density-Based Clustering (DBSCAN): Finds clusters in dense areas and treats scattered points as noise.
        - Mean-Shift Clustering: Discovers clusters by moving points toward the most crowded areas.
        - Spectral Clustering: Groups data by analyzing connections between points using graphs.
        """
        )
    elif(section == "Association"):    
        st.markdown("### Association")
        st.markdown("Association rule learning is a rule-based unsupervised learning technique used to discover interesting relationships between variables in large datasets. It identifies patterns in the form of “if-then” rules, showing how the presence of some items in the data implies the presence of others.")
        st.markdown(
            """
        - Finds frequent item combinations and the rules connecting them.
        - Commonly used in market basket analysis to understand product purchase relationships.
        - Helps retailers design promotions and cross-selling strategies.    
        """
        )
        st.markdown("## Some common Association algorithms:")
        st.markdown(
            """
        - Apriori Algorithm: Finds patterns by exploring frequent item combinations step-by-step.
        - FP-Growth Algorithm: An Efficient Alternative to Apriori. It quickly identifies frequent patterns without generating candidate sets.
        - Eclat Algorithm: Uses intersections of itemsets to efficiently find frequent patterns.
        - Efficient Tree-based Algorithms: Scales to handle large datasets by organizing data in tree structures.
        """
        )
    elif(section == "Dimensionality"):    
        st.markdown("### Dimensionality")
        st.markdown("Dimensionality reduction is the process of decreasing the number of features or variables in a dataset while retaining as much of the original information as possible. This technique helps simplify complex data making it easier to analyze and visualize. It also improves the efficiency and performance of machine learning algorithms by reducing noise and computational cost.")
        st.markdown(
            """
        - It reduces the dataset’s feature space from many dimensions to fewer, more meaningful ones.
        - Helps focus on the most important traits or patterns in the data.
        - Commonly used to improve model speed and reduce overfitting.    
        """
        )
        st.markdown("## Some common Dimensionality algorithms:")
        st.markdown(
            """
        - Principal Component Analysis (PCA): Reduces dimensions by transforming data into uncorrelated principal components.
        - Non-negative Matrix Factorization (NMF): Breaks data into non-negative parts to simplify representation.
        - Locally Linear Embedding (LLE): Reduces dimensions while preserving the relationships between nearby points.
        - Isomap: Captures global data structure by preserving distances along a manifold.
        """
        )
    elif(section == "Overview"):    
        st.markdown("### Advantages")
        st.markdown(
            """
        - Works with raw, unlabeled data, saving time and effort required for data annotation.
        - Finds hidden patterns and natural groupings in data that may not be easily identified by humans.
        - Handles large and complex datasets efficiently, including high-dimensional data.
        - Helps detect anomalies and unusual data points without needing prior examples.  
        """
        )
        st.markdown("## Challenges")
        st.markdown(
            """
        - Noisy data and outliers can distort patterns and reduce the effectiveness of the model.
        - Models may capture noise instead of meaningful patterns, leading to overfitting.
        - Lack of labeled data makes it difficult to guide the algorithm toward specific outcomes.
        - Results such as clusters may be difficult to interpret or may not clearly match real-world categories.
        """
        )    
if section == 'Overview':
    st.subheader('Choose a supervised learning algorithm')
    st.info('Select Regression or Classification from the sidebar to open an algorithm workspace.')
    for category, names in ALGORITHMS.items():
        with st.expander(f'{category} ({len(names)})', expanded=True):
            st.write(' · '.join(names))
else:
    st.subheader(algorithm)
    st.caption(f'{section} → {algorithm}')
    tab_data, tab_code = st.tabs(['📁 Dataset', '🧪 Implementation'])
    with tab_data:
        st.info(f"The dataset should be located at datasets/{section}/{section}_{algorithm.lower().replace(' ', '_')}.csv")
        st.write('Dataset workspace')
        
        uploadedFile = pd.read_csv(f'datasets/{section}/{section}_{algorithm.lower().replace(" ", "_")}.csv')
        st.dataframe(uploadedFile)
        st.caption('Read and process the file in your implementation module.')
    with tab_code:
        st.write('Algorithm workspace')
        # if st.button('Run algorithm', key=f'{section}_{algorithm}_run'):
        module_name = ''.join(c.lower() if c.isalnum() else '_' for c in algorithm).strip('_')
        package = section.lower()
        try:
            module = import_module(f'algorithms.{package}.{module_name}')
            module.run(section, algorithm)
        except ModuleNotFoundError as exc:
            if exc.name == f'algorithms.{package}.{module_name}':
                st.warning(f'Create algorithms/{package}/{module_name}.py with a run(uploadedFile) function.')
            else:
                st.exception(exc)
        except Exception as exc:
            st.exception(exc)

