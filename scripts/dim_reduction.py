import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import f_oneway, chi2_contingency, pearsonr
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import HDBSCAN
import prince
import os

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

def prepare_data():
    """Load and prepare all required data."""
    # Load imputed data
    data = np.load('../outputs/imputed_lora_religion.npy')
    data_indicator = np.load('../outputs/imputed_lora_indicators_religion.npy')
    
    # Load metadata
    raw_df = pd.read_feather("../data/data.feather")
    test_items_df = pd.read_csv("../data/test_items.csv", index_col=0)
    question_data = pd.read_csv("../data/question_data.csv", sep=';', index_col=0)
    question_weights = pd.read_csv("../outputs/question_weights.csv", index_col=0)
    
    # Generate multi_index for question-option pairs
    multi_index = []
    for col in raw_df[question_weights.index]:
        question_option_pairs = [(col, cat) for cat in raw_df[col].cat.categories]
        multi_index.extend(question_option_pairs)
    
    # Calculate intelligence proxy
    test_qs = [q for q in test_items_df.index if q in raw_df.columns]
    iq_answered = raw_df[test_qs].notna().sum(axis=1)
    
    scored_cols = []
    for q in test_qs:
        correct_answer = test_items_df.loc[q].option_correct
        correct = raw_df[q].cat.codes + 1 == correct_answer
        scored_cols.append(correct)
    
    total_score = pd.concat(scored_cols, axis=1).sum(axis=1)
    IQ_SCORES = total_score / iq_answered
    
    # Prepare metadata
    gender = raw_df['gender'] == 'Man'
    race = raw_df['race']
    num_answered = raw_df.isna().values.sum(axis=1)
    
    return {
        'data': data,
        'data_indicator': data_indicator,
        'raw_df': raw_df,
        'IQ_SCORES': IQ_SCORES,
        'gender': gender,
        'race': race,
        'num_answered': num_answered,
        'multi_index': multi_index  # Added to the returned dictionary
    }

def run_tsne_analysis(data_dict, important_qs, multi_index):
    """Run t-SNE analysis on important features."""
    indices_important = np.array([i for i, (col, cat) in enumerate(multi_index) if col in important_qs])
    VIP_data = data_dict['data'][:, indices_important]
    
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_result_VIP = tsne.fit_transform(VIP_data)
    
    return tsne_result_VIP

def visualize_tsne(tsne_result, color_data, title, color_label, cmap='viridis', filename=None):
    """Visualize t-SNE results with coloring."""
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        tsne_result[:, 0], 
        tsne_result[:, 1],
        c=color_data,
        cmap=cmap,
        alpha=0.7,
        s=3
    )
    plt.colorbar(scatter, label=color_label)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    if filename:
        plt.savefig(f"outputs/{filename}")
    plt.show()

def cluster_and_analyze(tsne_result, data_dict):
    """Perform clustering and statistical analysis."""
    # Cluster with HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=20)
    cluster_labels = clusterer.fit_predict(tsne_result)
    
    # ANOVA for IQ scores across clusters
    mask = (~np.isnan(data_dict['IQ_SCORES'])) & (cluster_labels != -1)
    valid_labels = cluster_labels[mask]
    valid_scores = data_dict['IQ_SCORES'][mask]
    
    groups = np.unique(valid_labels)
    iq_by_cluster = [valid_scores[valid_labels == g] for g in groups]
    f_stat, p_value = f_oneway(*iq_by_cluster)
    print(f"\nANOVA for IQ across clusters:")
    print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.3g}")
    
    # Chi-square test for race distribution across clusters
    mask = (data_dict['raw_df']['race'].notna().values) & (cluster_labels != -1)
    race_codes = data_dict['raw_df']['race'].cat.codes.values[mask]
    filtered_clusters = cluster_labels[mask]
    
    contingency = pd.crosstab(filtered_clusters, race_codes)
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square test for race distribution:")
    print(f"Chi-square: {chi2:.3f}, p-value: {p:.3g}")
    
    return cluster_labels

def run_mca_analysis(data_dict):
    """Perform MCA analysis and UMAP projection."""
    # Prepare one-hot encoded data (simplified version)
    # Note: The original notebook had more complex MCA preparation
    subsample = pd.DataFrame(data_dict['data']).sample(n=5000, random_state=42)
    zero_sum_cols = subsample.columns[subsample.sum(axis=0) == 0]
    subsample = subsample.drop(columns=zero_sum_cols)
    
    mca = prince.MCA(
        n_components=100,
        n_iter=3,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42,
        one_hot=False
    )
    mca_features = mca.fit_transform(subsample)
    
    # UMAP projection
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_features = umap.fit_transform(mca_features)
    
    return umap_features, mca_features.index

def analyze_correlation(data_dict):
    """Analyze correlation between IQ and number of questions answered."""
    mask = data_dict['IQ_SCORES'].notna() & (~np.isnan(data_dict['num_answered']))
    x = data_dict['IQ_SCORES'][mask].values
    y = data_dict['num_answered'][mask]
    
    r, p_value = pearsonr(x, y)
    print(f"\nPearson correlation between IQ and number answered:")
    print(f"r = {r:.3f}, p = {p_value:.3g}")
    
    # Plot
    def jitter(arr, frac=0.002):
        return arr + np.random.normal(0, frac * (np.max(arr) - np.min(arr)), arr.shape)
    
    plt.scatter(jitter(x), jitter(y), s=10, alpha=0.7)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, linestyle="--", linewidth=1, color="red", alpha=0.5)
    plt.xlabel("Intelligence Proxy")
    plt.ylabel("Number Answered")
    plt.title(f"Pearson r={r:.2f}, p={p_value:.2g}")
    plt.savefig('outputs/pearson_chart.pdf')
    plt.show()

def main():
    print("preparing data...")
    data_dict = prepare_data()
    
    important_qs = [
        'q21175', 'q41099', 'q45428', 'q44384', 
        'q60145', 'q29055', 'q18763'
    ]
    
    print("running tsne on top features...")
    tsne_result = run_tsne_analysis(data_dict, important_qs, data_dict['multi_index'])

    print("plotting graphs...")    
    visualize_tsne(tsne_result, data_dict['IQ_SCORES'], 
                 't-SNE Projection of Top Features', 
                 'Proxy Intelligence Score',
                 filename='top_iq.pdf')
    
    visualize_tsne(tsne_result, data_dict['raw_df']['race'].cat.codes, 
                 't-SNE Projection of Top Features', 
                 'Race', cmap='tab10',
                 filename='top_race.pdf')
    
    print("running clustering algorithm...")
    cluster_labels = cluster_and_analyze(tsne_result, data_dict)
    
    print("plotting tsne with clusters....")
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
               c=cluster_labels, cmap='tab20', alpha=0.7, s=1)
    plt.title('t-SNE Projection with Clusters', fontsize=16)
    plt.tight_layout()
    plt.savefig("outputs/top_clusters.pdf")
    plt.show()
    
    print("running umap on mca subset...")
    umap_features, umap_indices = run_mca_analysis(data_dict)
    
    # Visualize UMAP results
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_features[:, 0], umap_features[:, 1], 
               c=data_dict['IQ_SCORES'][umap_indices], cmap='viridis', 
               alpha=0.5, s=5)
    plt.colorbar(label='Proxy IQ score (normalized)')
    plt.title('UMAP of Imputed Data')
    plt.savefig("outputs/iq_umap.pdf")
    plt.show()

    print("correlation calculation...")    
    analyze_correlation(data_dict)

if __name__ == "__main__":
    main()