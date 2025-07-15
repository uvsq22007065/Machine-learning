import os
import pandas as pd
import matplotlib.pyplot as plt

# Dossier principal contenant les sous-dossiers des sujets
root_dir = "Reprise_24_mai_2025//results"

# Parcourt chaque sujet
for subject_id in range(1, 9):
    subject_folder = f"results_subject{subject_id}"
    subject_path = os.path.join(root_dir, subject_folder)

    if not os.path.exists(subject_path):
        print(f"Dossier {subject_path} introuvable.")
        continue

    # Parcourt les sous-sous-dossiers (modèles) de chaque sujet
    for subdir in os.listdir(subject_path):
        subdir_path = os.path.join(subject_path, subdir)
        if not os.path.isdir(subdir_path):
            continue

        svg_filename = f"subject{subject_id}_metrics_comparison.svg"
        svg_path = os.path.join(subdir_path, svg_filename)

        # Sauter si le graphique existe déjà
        if os.path.exists(svg_path):
            print(f"[✔] SVG trouvé pour {svg_path}")
            continue

        csv_filename = f"subject{subject_id}_overall_results.csv"
        csv_path = os.path.join(subdir_path, csv_filename)

        if not os.path.exists(csv_path):
            print(f"[⚠] CSV manquant : {csv_path}")
            continue

        # Lire les données
        df = pd.read_csv(csv_path)

        # Vérifie si les colonnes de métriques sont présentes
        metrics = ['mse', 'rmse', 'mae', 'r2']
        if not all(metric in df.columns for metric in metrics):
            print(f"[⚠] Colonnes de métriques manquantes dans {csv_path}")
            continue

        # Création du graphique
        plt.figure(figsize=(10, 6))
        for metric in metrics:
            plt.plot(df['data_percentage'], df[metric], label=metric.upper(), marker='o')

        plt.title(f'Subject {subject_id} - Comparaison des métriques')
        plt.xlabel('% des données utilisées pour l’entraînement')
        plt.ylabel('Valeur')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Sauvegarde
        plt.savefig(svg_path)
        plt.close()
        print(f"[+] Graphique créé : {svg_path}")
