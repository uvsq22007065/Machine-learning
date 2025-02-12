import scipy.io
import numpy as np
import os
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import ListedColormap  # Ajout de l'import

import os
import numpy as np
import scipy.io

# Fonction pour simplifier les noms de fichiers (extraction des chiffres)
def simplify_filename(filename):
    """Extract numeric or short identifiers from filenames."""
    return ''.join([c for c in filename if c.isdigit()])  # Garde uniquement les chiffres

def plot_correlation_heatmap(results, group_name, correlation_threshold):
    """Plot a heatmap of correlations between all compared files."""
    # Simplifier les noms des fichiers
    files = sorted(list(set(
        [simplify_filename(os.path.basename(result['file1'])) for result in results] +
        [simplify_filename(os.path.basename(result['file2'])) for result in results]
    )))
    n = len(files)
    
    # Créer une matrice de corrélation remplie de NaN
    correlation_matrix = np.full((n, n), np.nan)
    
    # Remplir la matrice avec les valeurs de corrélation
    for result in results:
        file1 = simplify_filename(os.path.basename(result['file1']))
        file2 = simplify_filename(os.path.basename(result['file2']))
        i, j = files.index(file1), files.index(file2)
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                correlation_matrix[i, j] = metrics['correlation']
                correlation_matrix[j, i] = metrics['correlation']  # Symétrie

    # Créer un colormap personnalisé
    coolwarm_cmap = ListedColormap(plt.cm.coolwarm(np.linspace(0, 1, 256)))

    # Tracer la heatmap avec annotations uniquement pour les valeurs significatives
    plt.figure(figsize=(8, 6))
    mask = np.isnan(correlation_matrix)  # Masque pour les NaN
    sns.heatmap(correlation_matrix, xticklabels=files, yticklabels=files,
                annot=True, fmt=".2f", mask=mask, cmap=coolwarm_cmap, cbar_kws={'label': 'Correlation'},
                annot_kws={"size": 8})
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title(f"Correlation Heatmap for {group_name} (Threshold ≥ {correlation_threshold})")
    plt.tight_layout()
    plt.show()

def plot_correlation_graph(results, group_name, correlation_threshold):
    """Plot a graph where nodes are files and edges represent high correlation."""
    G = nx.Graph()
    
    # Ajouter des arêtes pour les paires avec forte corrélation
    for result in results:
        file1 = simplify_filename(os.path.basename(result['file1']))
        file2 = simplify_filename(os.path.basename(result['file2']))
        for key, metrics in result['metrics'].items():
            if abs(metrics['correlation']) >= correlation_threshold:
                G.add_edge(file1, file2, weight=metrics['correlation'], label=f"{metrics['correlation']:.2f}")
    
    # Dessiner le graphe
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)  # Disposition des nœuds
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue",
            font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(f"Correlation Graph for {group_name} (Threshold ≥ {correlation_threshold})")
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
        #print(f"Error loading {file_path}: {e}")
        return None

def compare_signals(signal1, signal2):
    """Compare two signals and return similarity metrics."""
    min_length = min(len(signal1), len(signal2))
    signal1, signal2 = signal1[:min_length], signal2[:min_length]
    
    # Calculate similarity metrics
    mse = np.mean((signal1 - signal2) ** 2)
    correlation = np.corrcoef(signal1, signal2)[0, 1]
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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_correlation_bubble_3d(results, group_name, correlation_threshold):
    """
    Plot a 3D bubble chart where nodes are files, and bubble sizes represent correlations.
    """
    # Simplifier les noms de fichiers
    files = sorted(list(set(
        [simplify_filename(os.path.basename(result['file1'])) for result in results] +
        [simplify_filename(os.path.basename(result['file2'])) for result in results]
    )))
    n = len(files)

    # Coordonnées des fichiers dans l'espace 3D
    coords = {file: (np.random.rand(), np.random.rand(), np.random.rand()) for file in files}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ajouter des points pour chaque fichier
    for file, (x, y, z) in coords.items():
        ax.scatter(x, y, z, color="skyblue", s=200, label=file)
    
    # Tracer les bulles représentant les corrélations
    for result in results:
        file1 = simplify_filename(os.path.basename(result['file1']))
        file2 = simplify_filename(os.path.basename(result['file2']))
        if file1 in coords and file2 in coords:
            for key, metrics in result['metrics'].items():
                if abs(metrics['correlation']) >= correlation_threshold:
                    x1, y1, z1 = coords[file1]
                    x2, y2, z2 = coords[file2]
                    
                    # Calcul de la position de la bulle au milieu des deux nœuds
                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2
                    z_mid = (z1 + z2) / 2
                    
                    # Taille de la bulle proportionnelle à la corrélation
                    bubble_size = abs(metrics['correlation']) * 1000
                    
                    ax.scatter(x_mid, y_mid, z_mid, s=bubble_size, alpha=0.5,
                               color="orange", edgecolor="k")
                    
                    # Ajouter un texte pour indiquer la corrélation
                    ax.text(x_mid, y_mid, z_mid, f"{metrics['correlation']:.2f}", fontsize=8)

    # Configurer les axes
    ax.set_title(f"3D Bubble Chart for {group_name} (Threshold ≥ {correlation_threshold})", fontsize=14)
    ax.set_xlabel('X-axis', fontsize=10)
    ax.set_ylabel('Y-axis', fontsize=10)
    ax.set_zlabel('Z-axis', fontsize=10)

    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_correlation_bubble_3d_with_links(results, group_name, correlation_threshold):
    """
    Plot a 3D bubble chart with explicit links where nodes are files and edges represent correlations.
    """
    # Simplifier les noms de fichiers
    def simplify_filename(filename):
        return filename.split('_')[0]  # Réduction au sujet uniquement (ex. "Subject1")

    files = sorted(list(set(
        [simplify_filename(os.path.basename(result['file1'])) for result in results] +
        [simplify_filename(os.path.basename(result['file2'])) for result in results]
    )))
    n = len(files)

    # Coordonnées aléatoires pour chaque fichier dans l'espace 3D
    np.random.seed(42)  # Fixer la graine pour des positions reproductibles
    coords = {file: (np.random.rand(), np.random.rand(), np.random.rand()) for file in files}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ajouter des points pour chaque fichier
    for file, (x, y, z) in coords.items():
        ax.scatter(x, y, z, color="skyblue", s=200, label=file)

    # Tracer les lignes et bulles pour représenter les corrélations
    for result in results:
        file1 = simplify_filename(os.path.basename(result['file1']))
        file2 = simplify_filename(os.path.basename(result['file2']))
        if file1 in coords and file2 in coords:
            for key, metrics in result['metrics'].items():
                if abs(metrics['correlation']) >= correlation_threshold:
                    x1, y1, z1 = coords[file1]
                    x2, y2, z2 = coords[file2]
                    
                    # Lien explicite (ligne) entre les deux fichiers
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", linewidth=1, alpha=0.7)
                    
                    # Position de la bulle au milieu des deux fichiers
                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2
                    z_mid = (z1 + z2) / 2
                    
                    # Taille de la bulle proportionnelle à la corrélation
                    bubble_size = abs(metrics['correlation']) * 1000
                    
                    ax.scatter(x_mid, y_mid, z_mid, s=bubble_size, alpha=0.6,
                               color="orange", edgecolor="k")
                    
                    # Annotation de la corrélation
                    ax.text(x_mid, y_mid, z_mid, f"{metrics['correlation']:.2f}", fontsize=8, color="black")

    # Configurer les axes
    ax.set_title(f"3D Correlation Bubble Chart for {group_name} (Threshold ≥ {correlation_threshold})", fontsize=14)
    ax.set_xlabel('X-axis', fontsize=10)
    ax.set_ylabel('Y-axis', fontsize=10)
    ax.set_zlabel('Z-axis', fontsize=10)

    plt.legend(loc="upper left", fontsize=8, title="Subjects")
    plt.tight_layout()
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm

def plot_correlation_bubble_3d_with_links_and_numbers(results, group_name, correlation_threshold):
    """
    Plot a 3D bubble chart with explicit links and numbered bubbles where nodes are files
    and edges represent correlations.
    """
    # Simplifier les noms de fichiers
    def simplify_filename(filename):
        return filename.split('_')[0]  # Réduction au sujet uniquement (ex. "Subject1")

    files = sorted(list(set(
        [simplify_filename(os.path.basename(result['file1'])) for result in results] +
        [simplify_filename(os.path.basename(result['file2'])) for result in results]
    )))
    n = len(files)

    # Coordonnées aléatoires pour chaque fichier dans l'espace 3D
    np.random.seed(42)  # Fixer la graine pour des positions reproductibles
    coords = {file: (np.random.rand(), np.random.rand(), np.random.rand()) for file in files}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ajouter des points pour chaque fichier
    for file, (x, y, z) in coords.items():
        ax.scatter(x, y, z, color="skyblue", s=200, label=file)

    # Tracer les liens et bulles pour représenter les corrélations
    colormap = cm.get_cmap("Reds")  # Colormap pour intensifier la couleur des liens
    bubble_count = 1  # Compteur pour numéroter les bulles

    for result in results:
        file1 = simplify_filename(os.path.basename(result['file1']))
        file2 = simplify_filename(os.path.basename(result['file2']))
        if file1 in coords and file2 in coords:
            for key, metrics in result['metrics'].items():
                if abs(metrics['correlation']) >= correlation_threshold:
                    x1, y1, z1 = coords[file1]
                    x2, y2, z2 = coords[file2]

                    # Intensité de la couleur du lien en fonction de la corrélation
                    intensity = abs(metrics['correlation'])
                    link_color = colormap(intensity)

                    # Lien explicite (ligne) entre les deux fichiers
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color=link_color, linewidth=2, alpha=0.8)

                    # Position de la bulle au milieu des deux fichiers
                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2
                    z_mid = (z1 + z2) / 2

                    # Taille de la bulle proportionnelle à la corrélation
                    bubble_size = abs(metrics['correlation']) * 1000

                    ax.scatter(x_mid, y_mid, z_mid, s=bubble_size, alpha=0.6,
                               color="orange", edgecolor="k")

                    # Ajouter un numéro sur la bulle
                    ax.text(x_mid, y_mid, z_mid, f"{bubble_count}", fontsize=10, color="black", ha='center')

                    # Légende pour corrélation
                    ax.text(x_mid, y_mid, z_mid - 0.05, f"{metrics['correlation']:.2f}", fontsize=8, color="gray", ha='center')
                    
                    bubble_count += 1

    # Configurer les axes
    ax.set_title(f"3D Correlation Bubble Chart for {group_name} (Threshold ≥ {correlation_threshold})", fontsize=14)
    ax.set_xlabel('X-axis', fontsize=10)
    ax.set_ylabel('Y-axis', fontsize=10)
    ax.set_zlabel('Z-axis', fontsize=10)

    plt.legend(loc="upper left", fontsize=8, title="Subjects")
    plt.tight_layout()
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm

def plot_correlation_bubble_3d_optimized(results, group_name, correlation_threshold):
    """
    Plot an optimized 3D bubble chart with clearer visualization for nodes, links, and annotations.
    """
    # Simplifier les noms de fichiers
    def simplify_filename(filename):
        return filename.split('_')[0]

    files = sorted(list(set(
        [simplify_filename(os.path.basename(result['file1'])) for result in results] +
        [simplify_filename(os.path.basename(result['file2'])) for result in results]
    )))
    n = len(files)

    # Coordonnées aléatoires pour chaque fichier dans l'espace 3D
    np.random.seed(42)
    coords = {file: (np.random.rand(), np.random.rand(), np.random.rand()) for file in files}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ajouter des points pour chaque fichier
    for file, (x, y, z) in coords.items():
        ax.scatter(x, y, z, color="skyblue", s=100, label=file, alpha=0.9, edgecolor='k')

    # Tracer les liens et bulles pour représenter les corrélations
    colormap = cm.get_cmap("Reds")
    bubble_count = 1  # Numéro des bulles

    for result in results:
        file1 = simplify_filename(os.path.basename(result['file1']))
        file2 = simplify_filename(os.path.basename(result['file2']))
        if file1 in coords and file2 in coords:
            for key, metrics in result['metrics'].items():
                if abs(metrics['correlation']) >= correlation_threshold:
                    x1, y1, z1 = coords[file1]
                    x2, y2, z2 = coords[file2]

                    # Intensité de la couleur pour les liens
                    intensity = abs(metrics['correlation'])
                    link_color = colormap(intensity)

                    # Lien avec transparence
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color=link_color, linewidth=2, alpha=0.7)

                    # Position de la bulle
                    x_mid = (x1 + x2) / 2
                    y_mid = (y1 + y2) / 2
                    z_mid = (z1 + z2) / 2

                    # Taille réduite des bulles
                    bubble_size = abs(metrics['correlation']) * 500
                    ax.scatter(x_mid, y_mid, z_mid, s=bubble_size, alpha=0.5,
                               color="orange", edgecolor="k")

                    # Affichage du numéro uniquement
                    ax.text(x_mid, y_mid, z_mid, f"{bubble_count}", fontsize=9, color="black", ha='center')
                    bubble_count += 1

    # Configurer les axes
    ax.set_title(f"3D Optimized Correlation Bubble Chart for {group_name} (Threshold ≥ {correlation_threshold})", fontsize=12)
    ax.set_xlabel('X-axis', fontsize=10)
    ax.set_ylabel('Y-axis', fontsize=10)
    ax.set_zlabel('Z-axis', fontsize=10)

    plt.tight_layout()
    plt.show()

def lowest_pair(results):
    """
    Identifie la paire de fichiers avec la corrélation la plus basse et affiche les signaux vGRF et angle.
    """
    lowest_correlation = float('inf')  # Initialise avec une valeur très élevée
    lowest_pair_result = None

    # Parcourir les résultats pour trouver la plus petite corrélation
    for result in results:
        for key, metrics in result['metrics'].items():
            if key == 'vGRF':  # Utilise 'vGRF' comme référence pour trouver la plus faible corrélation
                if metrics['correlation'] < lowest_correlation:
                    lowest_correlation = metrics['correlation']
                    lowest_pair_result = result

    # Vérification si une paire a été trouvée
    if lowest_pair_result is None:
        print("Aucune paire trouvée avec des résultats valides.")
        return

    # Charger les données des fichiers correspondants
    file1 = lowest_pair_result['file1']
    file2 = lowest_pair_result['file2']
    print(f"Paire avec la corrélation la plus basse ({lowest_correlation:.2f}):")
    print(f" - {os.path.basename(file1)}")
    print(f" - {os.path.basename(file2)}")

    data1 = load_mat_file(file1)
    data2 = load_mat_file(file2)

    # Tracer les courbes vGRF et angle pour les deux fichiers
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Tracer vGRF
    axes[0].plot(data1['vGRF'], label=f"{simplify_filename(os.path.basename(file1))} vGRF", color='blue')
    axes[0].plot(data2['vGRF'], label=f"{simplify_filename(os.path.basename(file2))} vGRF", color='red')
    axes[0].set_title("vGRF Comparison")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("vGRF")
    axes[0].legend()

    # Tracer Angle
    axes[1].plot(data1['angle'], label=f"{simplify_filename(os.path.basename(file1))} angle", color='blue')
    axes[1].plot(data2['angle'], label=f"{simplify_filename(os.path.basename(file2))} angle", color='red')
    axes[1].set_title("Angle Comparison")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Angle")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = "C:\\Users\\Grégoire\\OneDrive\\Bureau\\EPF\\BRL\\Machine learning\\GaitPhaseEstimatorData\\dataHealthy[1]"  # Replace with your directory

    # Comparaison des fichiers left et right
    left_results, right_results = process_all_subjects(data_dir)
    
    # Heatmap pour les fichiers left et right
    plot_correlation_heatmap(left_results, "Left files", correlation_threshold=0.98)
    plot_correlation_heatmap(right_results, "Right files", correlation_threshold=0.98)
    
    # Graphe pour les fichiers left et right
    plot_correlation_graph(left_results, "Left files", correlation_threshold=0.98)
    plot_correlation_graph(right_results, "Right files", correlation_threshold=0.98)

    # Graphe 3D pour les fichiers left et right
    # plot_correlation_bubble_3d(left_results, "Left files", correlation_threshold=0.99)
    # plot_correlation_bubble_3d(right_results, "Right files", correlation_threshold=0.99)

    # Graphe 3D avec liens pour les fichiers left et right
    # plot_correlation_bubble_3d_with_links(left_results, "Left files", correlation_threshold=0.99)
    # plot_correlation_bubble_3d_with_links(right_results, "Right files", correlation_threshold=0.99)

    # Graphe 3D avec liens pour les fichiers left et right
    # plot_correlation_bubble_3d_with_links_and_numbers(left_results, "Left files", correlation_threshold=0.99)
    # plot_correlation_bubble_3d_with_links_and_numbers(right_results, "Right files", correlation_threshold=0.99)

    # plot_correlation_bubble_3d_optimized(left_results, "Left files", correlation_threshold=0.99)
    # plot_correlation_bubble_3d_optimized(right_results, "Right files", correlation_threshold=0.99)

    # Trouver la paire avec la plus faible corrélation pour chaque groupe
    lowest_left = lowest_pair(left_results)
    lowest_right = lowest_pair(right_results)

    # Afficher les résultats
    print("Lowest correlation in LEFT group:")
    print(lowest_left)

    print("Lowest correlation in RIGHT group:")
    print(lowest_right)