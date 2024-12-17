import scipy.io
import numpy as np
import os
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import ListedColormap  # Ajout de l'import

def plot_correlation_heatmap(results, group_name, correlation_threshold=0.5):
    """Plot a heatmap of correlations between all compared files."""
    files = sorted(list(set(
        [os.path.basename(result['file1']) for result in results] +
        [os.path.basename(result['file2']) for result in results]
    )))
    n = len(files)
    
    # Créer une matrice de corrélation remplie de NaN
    correlation_matrix = np.full((n, n), np.nan)
    
    # Remplir la matrice avec les valeurs de corrélation
    for result in results:
        file1 = os.path.basename(result['file1'])
        file2 = os.path.basename(result['file2'])
        i, j = files.index(file1), files.index(file2)
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                correlation_matrix[i, j] = metrics['correlation']
                correlation_matrix[j, i] = metrics['correlation']  # Symétrie

    # Créer un colormap personnalisé en utilisant ListedColormap
    coolwarm_cmap = ListedColormap(plt.cm.coolwarm(np.linspace(0, 1, 256)))

    # Tracer la heatmap avec le colormap personnalisé
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, xticklabels=files, yticklabels=files, annot=True, fmt=".2f", cmap=coolwarm_cmap, cbar_kws={'label': 'Correlation'})
    plt.title(f"Correlation Heatmap for {group_name}")
    plt.tight_layout()
    plt.show()

def plot_correlation_graph(results, group_name, correlation_threshold=0.9):
    """Plot a graph where nodes are files and edges represent high correlation."""
    G = nx.Graph()
    
    # Ajouter des arêtes pour les paires avec forte corrélation
    for result in results:
        file1 = os.path.basename(result['file1'])
        file2 = os.path.basename(result['file2'])
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                G.add_edge(file1, file2, weight=metrics['correlation'], label=f"{metrics['correlation']:.2f}")
    
    # Dessiner le graphe
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)  # Disposition des nœuds
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"Correlation Graph for {group_name} (Threshold >= {correlation_threshold})")
    plt.show()

def load_mat_file(file_path):
    """Load .mat file and return relevant keys."""
    try:
        mat_data = scipy.io.loadmat(file_path)
        return {
            'vGRF': mat_data['vGRF'][0, 0][0][0],  # Ground Reaction Force
            'angle': mat_data['angle'][0, 0][0][0],  # Ankle Angle
            'torque': mat_data['torque'][0, 0][0][0],  # Torque
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_signals(signal1, signal2):
    """Compare two signals and return similarity metrics."""
    # Ensure signals have the same length for comparison
    min_length = min(len(signal1), len(signal2))
    signal1, signal2 = signal1[:min_length], signal2[:min_length]
    
    # Calculate similarity metrics
    mse = np.mean((signal1 - signal2) ** 2)  # Mean Squared Error
    correlation = np.corrcoef(signal1, signal2)[0, 1]  # Correlation Coefficient
    return mse, correlation

def compare_group(files, data_directory):
    """Compare all files within a group (_left or _right) and return comparison results."""
    results = []
    file_paths = [os.path.join(data_directory, f) for f in files]
    
    for file1, file2 in combinations(file_paths, 2):
        data1 = load_mat_file(file1)
        data2 = load_mat_file(file2)
        
        if data1 is None or data2 is None:
            continue

        comparison = {'file1': file1, 'file2': file2, 'metrics': {}}
        for key in ['vGRF', 'angle', 'torque']:
            if key in data1 and key in data2:
                mse, correlation = compare_signals(data1[key], data2[key])
                comparison['metrics'][key] = {'mse': mse, 'correlation': correlation}
        
        results.append(comparison)
    
    return results

def process_all_subjects(data_directory):
    """Compare all _left and _right files in the directory."""
    files = os.listdir(data_directory)
    left_files = sorted([f for f in files if '_left.mat' in f])
    right_files = sorted([f for f in files if '_right.mat' in f])

    print("Comparing all _left.mat files...")
    left_results = compare_group(left_files, data_directory)

    print("Comparing all _right.mat files...")
    right_results = compare_group(right_files, data_directory)

    return left_results, right_results

def print_results(results, group_name, correlation_threshold=0.9):
    """Pretty print the comparison results for highly correlated pairs."""
    print(f"Highly Correlated Results for {group_name} (Threshold >= {correlation_threshold}):")
    for result in results:
        has_high_correlation = False  # Flag to determine if at least one key meets the threshold
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                if not has_high_correlation:
                    print(f"  Comparing {os.path.basename(result['file1'])} and {os.path.basename(result['file2'])}:")
                    has_high_correlation = True  # Print the pair only once
                print(f"    {key}: MSE={metrics['mse']:.4f}, Correlation={metrics['correlation']:.4f}")
    print("\n")

def filter_valid_pairs(left_results, right_results, correlation_threshold=0.9):
    """Filter pairs where both right and left files have high correlation."""
    valid_pairs = []
    
    # Créer un dictionnaire pour les résultats des fichiers _left
    left_correlation_dict = {}
    for result in left_results:
        file1 = os.path.basename(result['file1']).replace("_left.mat", "")
        file2 = os.path.basename(result['file2']).replace("_left.mat", "")
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                left_correlation_dict[(file1, file2)] = metrics['correlation']

    # Vérifier les paires _right et voir si leurs homonymes _left sont aussi corrélés
    for result in right_results:
        file1 = os.path.basename(result['file1']).replace("_right.mat", "")
        file2 = os.path.basename(result['file2']).replace("_right.mat", "")
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                if (file1, file2) in left_correlation_dict:
                    # Ajouter les fichiers si _right et _left corrèlent bien
                    valid_pairs.append({
                        'pair': (f"{file1}_left.mat", f"{file2}_left.mat", f"{file1}_right.mat", f"{file2}_right.mat"),
                        'right_correlation': metrics['correlation'],
                        'left_correlation': left_correlation_dict[(file1, file2)],
                        'key': key
                    })
    return valid_pairs

if __name__ == "__main__":
    data_dir = "C:\\Users\\Grégoire\\OneDrive\\Bureau\\EPF\\BRL\\Machine learning\\GaitPhaseEstimatorData\\dataHealthy[1]"  # Replace with your directory
    
    # Comparaison des fichiers left et right
    left_results, right_results = process_all_subjects(data_dir)
    
    # Heatmap pour les fichiers left et right
    plot_correlation_heatmap(left_results, "Left files", correlation_threshold=0.5)
    plot_correlation_heatmap(right_results, "Right files", correlation_threshold=0.5)
    
    # Graphe pour les fichiers left et right
    plot_correlation_graph(left_results, "Left files", correlation_threshold=0.9)
    plot_correlation_graph(right_results, "Right files", correlation_threshold=0.9)

    # Afficher les résultats des paires valides
    print("Paires valides avec bonne corrélation pour left et right:")
    for pair in valid_pairs:
        print(f"  {pair['pair'][0]} & {pair['pair'][1]} corrèlent bien avec {pair['pair'][2]} & {pair['pair'][3]}")
        print(f"    Signal: {pair['key']}, Corrélation Right: {pair['right_correlation']:.4f}, Corrélation Left: {pair['left_correlation']:.4f}")
