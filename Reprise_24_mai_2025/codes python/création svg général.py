import os
import pandas as pd
import matplotlib.pyplot as plt

# Dossier principal
root_dir = "Reprise_24_mai_2025//results"
metrics = ['mse', 'rmse', 'mae', 'r2']

for subject_id in range(1, 2):
    subject_folder = f"results_subject{subject_id}"
    subject_path = os.path.join(root_dir, subject_folder)

    if not os.path.exists(subject_path):
        print(f"[!] Dossier {subject_path} introuvable.")
        continue

    metric_data = {metric: {} for metric in metrics}

    for model_folder in os.listdir(subject_path):
        model_path = os.path.join(subject_path, model_folder)
        if not os.path.isdir(model_path):
            continue

        csv_filename = f"subject{subject_id}_overall_results.csv"
        csv_path = os.path.join(model_path, csv_filename)

        if not os.path.exists(csv_path):
            print(f"[⚠] CSV manquant : {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            df.columns = [col.strip() for col in df.columns]  # Nettoyer les noms de colonnes
        except Exception as e:
            print(f"[‼] Erreur lecture {csv_path} : {e}")
            continue

        if 'data_percentage' not in df.columns or not all(metric in df.columns for metric in metrics):
            print(f"[⚠] Colonnes manquantes dans {csv_path}")
            continue

        # === Cas Classical ===
        if "Classical" in model_folder and 'model' in df.columns:
            for model_name in df['model'].unique():
                model_name_clean = model_name.strip() + " (classical)"
                sub_df = df[df['model'] == model_name]
                for metric in metrics:
                    metric_data[metric][model_name_clean] = (
                        sub_df['data_percentage'], sub_df[metric]
                    )
        else:
            # Cas général
            label = model_folder.split('_')[0].strip()
            for metric in metrics:
                metric_data[metric][label] = (
                    df['data_percentage'], df[metric]
                )

    # === Générer un fichier par métrique ===
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        for model_name, (x_vals, y_vals) in metric_data[metric].items():
            plt.plot(x_vals, y_vals, label=model_name, marker='o')

        plt.title(f"Subject {subject_id} - {metric.upper()}")
        plt.xlabel('% des données utilisées')
        plt.ylabel(metric.upper())
        plt.grid(True)
        plt.legend(fontsize='small')
        plt.tight_layout()

        output_path = os.path.join(subject_path, f"subject{subject_id}_{metric}_comparison.svg")
        plt.savefig(output_path)
