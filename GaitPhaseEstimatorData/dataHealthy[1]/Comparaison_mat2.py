import scipy.io
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
from itertools import combinations

def interpolate_data(data, time, target_points):
    """
    Interpole les données sur un nombre cible de points.
    """
    if time.size == 1:
        time = np.linspace(0, 1, data.size)
    else:
        time = time.squeeze()
    
    new_time = np.linspace(0, 1, target_points)
    interpolated_data = np.interp(new_time, time, data.squeeze())
    return interpolated_data, new_time

def process_file(file_path, output_directory, target_points=26000, keys=['vGRF', 'angle']):
    """
    Traite un fichier .mat et sauvegarde les données interpolées.
    """
    data = scipy.io.loadmat(file_path)
    interpolated_data = {}
    
    for key in keys:
        if key in data:
            structure = data[key]
            if isinstance(structure, np.ndarray) and structure.size == 1 and structure.dtype.names:
                if 'Data' in structure.dtype.names and 'Time' in structure.dtype.names:
                    original_data = structure['Data'][0, 0]
                    original_time = structure['Time'][0, 0]
                    interpolated_data[key] = {}
                    interpolated_data[key]['Data'], interpolated_data[key]['Time'] = interpolate_data(
                        original_data, original_time, target_points
                    )
                else:
                    print(f"Attention : 'Data' ou 'Time' manquant dans la structure {key}")
            else:
                print(f"Structure invalide pour la clé : {key}")
    
    output_path = os.path.join(output_directory, os.path.basename(file_path))
    scipy.io.savemat(output_path, {key: np.array([(interpolated_data[key]['Data'], interpolated_data[key]['Time'])],
                                                 dtype=[('Data', 'O'), ('Time', 'O')]) for key in interpolated_data})
    print(f"Fichier sauvegardé : {output_path}")

def process_directory(input_directory, output_directory, target_points=26000, keys=['vGRF', 'angle']):
    """
    Traite tous les fichiers .mat dans un répertoire.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.mat'):
            file_path = os.path.join(input_directory, file_name)
            process_file(file_path, output_directory, target_points, keys)

def compare_signals(signal1, signal2):
    """
    Compare deux signaux et retourne la MSE et la corrélation.
    Gère les cas où les données ne sont pas valides.
    """
    # Vérifier si les deux signaux sont des séquences
    if not (isinstance(signal1, (np.ndarray, list)) and isinstance(signal2, (np.ndarray, list))):
        raise ValueError(f"Les signaux ne sont pas valides pour la comparaison : types {type(signal1)}, {type(signal2)}")
    
    # S'assurer que les signaux ont la même longueur
    min_length = min(len(signal1), len(signal2))
    signal1, signal2 = signal1[:min_length], signal2[:min_length]
    
    # Calcul de la MSE et de la corrélation
    mse = np.mean((signal1 - signal2) ** 2)
    correlation = np.corrcoef(signal1, signal2)[0, 1]
    return mse, correlation

def load_mat_file(file_path, keys=['vGRF', 'angle']):
    """
    Charge les données spécifiques depuis un fichier .mat.
    Gère les cas où les données sont mal formatées.
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        data = {}
        for key in keys:
            if key in mat_data:
                # Extraction des données
                raw_data = mat_data[key]
                if isinstance(raw_data, np.ndarray) and raw_data.size == 1 and raw_data.dtype.names:
                    if 'Data' in raw_data.dtype.names:
                        data[key] = raw_data['Data'][0, 0]
                    else:
                        print(f"Attention : 'Data' manquant pour la clé {key} dans {file_path}")
                else:
                    print(f"Structure inattendue pour la clé {key} dans {file_path}")
        return data
    except Exception as e:
        print(f"Erreur de chargement du fichier {file_path} : {e}")
        return None

def compare_group(files, data_directory, keys=['vGRF', 'angle']):
    """
    Compare tous les fichiers d'un groupe (ex. _left.mat ou _right.mat).
    Gère les cas où les données sont mal formatées.
    """
    results = []
    file_paths = [os.path.join(data_directory, f) for f in files]
    
    for file1, file2 in combinations(file_paths, 2):
        data1 = load_mat_file(file1, keys)
        data2 = load_mat_file(file2, keys)
        if data1 is None or data2 is None:
            print(f"Impossible de comparer {file1} et {file2} : données non valides")
            continue
        
        comparison = {'file1': file1, 'file2': file2, 'metrics': {}}
        for key in keys:
            if key in data1 and key in data2:
                try:
                    mse, correlation = compare_signals(data1[key], data2[key])
                    comparison['metrics'][key] = {'mse': mse, 'correlation': correlation}
                except ValueError as ve:
                    print(f"Erreur lors de la comparaison {file1} et {file2} pour la clé {key} : {ve}")
        results.append(comparison)
    return results

def plot_correlation_heatmap(results, group_name, correlation_threshold):
    """
    Trace une heatmap des corrélations.
    """
    files = sorted(list(set(
        [os.path.basename(result['file1']) for result in results] +
        [os.path.basename(result['file2']) for result in results]
    )))
    n = len(files)
    correlation_matrix = np.full((n, n), np.nan)
    for result in results:
        file1 = os.path.basename(result['file1'])
        file2 = os.path.basename(result['file2'])
        i, j = files.index(file1), files.index(file2)
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                correlation_matrix[i, j] = metrics['correlation']
                correlation_matrix[j, i] = metrics['correlation']
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, xticklabels=files, yticklabels=files, annot=True,
                fmt=".2f", mask=np.isnan(correlation_matrix), cmap="coolwarm", cbar_kws={'label': 'Correlation'})
    plt.xticks(rotation=45)
    plt.title(f"Heatmap de corrélation ({group_name}) (Seuil ≥ {correlation_threshold})")
    plt.tight_layout()
    plt.show()

def plot_correlation_graph(results, group_name, correlation_threshold):
    """
    Trace un graphe des corrélations.
    """
    G = nx.Graph()
    for result in results:
        file1 = os.path.basename(result['file1'])
        file2 = os.path.basename(result['file2'])
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                G.add_edge(file1, file2, weight=metrics['correlation'], label=f"{metrics['correlation']:.2f}")    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(f"Graphe de corrélation ({group_name}) (Seuil ≥ {correlation_threshold})")
    plt.show()

# Pipeline principale
if __name__ == "__main__":
    input_directory = "C:\\Users\\Grégoire\\OneDrive\\Bureau\\EPF\\BRL\\Machine learning\\GaitPhaseEstimatorData\\dataHealthy[1]"
    output_directory = "C:\\Users\\Grégoire\\OneDrive\\Bureau\\EPF\\BRL\\Machine learning\\GaitPhaseEstimatorData\\dataHealthy[1]\\interpolated"
    target_points = 26000

    print("Début de l'interpolation...")
    process_directory(input_directory, output_directory, target_points)

    print("Comparaison des fichiers interpolés...")
    files = os.listdir(output_directory)
    left_files = sorted([f for f in files if '_left.mat' in f])
    right_files = sorted([f for f in files if '_right.mat' in f])

    left_results = compare_group(left_files, output_directory)
    right_results = compare_group(right_files, output_directory)

    print("Génération des heatmaps et graphes...")
    plot_correlation_heatmap(left_results, "Fichiers LEFT", correlation_threshold=0.1)
    plot_correlation_heatmap(right_results, "Fichiers RIGHT", correlation_threshold=0.1)
    plot_correlation_graph(left_results, "Fichiers LEFT", correlation_threshold=0.1)
    plot_correlation_graph(right_results, "Fichiers RIGHT", correlation_threshold=0.1)
