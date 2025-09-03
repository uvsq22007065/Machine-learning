import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
root_dir = "Reprise_24_mai_2025//results"
metrics = ['mse', 'rmse', 'mae', 'r2']
all_data = []

# Étape 1 : Charger tous les CSV et agréger dans une seule table
for subject_id in range(1, 12):
    subject_folder = f"results_subject{subject_id}"
    subject_path = os.path.join(root_dir, subject_folder)

    if not os.path.exists(subject_path):
        continue

    for model_folder in os.listdir(subject_path):
        model_path = os.path.join(subject_path, model_folder)
        if not os.path.isdir(model_path):
            continue

        csv_file = os.path.join(model_path, f"subject{subject_id}_overall_results.csv")
        if not os.path.exists(csv_file):
            continue

        try:
            df = pd.read_csv(csv_file)
            df.columns = [col.strip() for col in df.columns]
        except Exception as e:
            print(f"[‼] Erreur dans {csv_file} : {e}")
            continue

        # Cas 'Classical' → utilise la colonne "model"
        if "Classical" in model_folder and 'model' in df.columns:
            pass  # modèles sont déjà différenciés dans la colonne 'model'
        else:
            # Autres cas → ajouter le nom du dossier comme modèle
            model_name = model_folder.split('_')[0].strip()
            df['model'] = model_name

        df['subject'] = f"subject{subject_id}"  # optionnel pour traçage par sujet si besoin
        all_data.append(df)

# Étape 2 : Concaténer tous les fichiers
all_df = pd.concat(all_data, ignore_index=True)
# === Filtrer certains modèles à ignorer ===
models_to_ignore = ['RNNWS', 'LSTMWS']
all_df = all_df[~all_df['model'].isin(models_to_ignore)]


# Étape 3 : Moyenne et écart-type par modèle et pourcentage
for metric in metrics:
    plt.figure(figsize=(8, 6))

    grouped = all_df.groupby(['model', 'data_percentage'])[metric].agg(['mean', 'std']).reset_index()

    for model_name in grouped['model'].unique():
        model_data = grouped[grouped['model'] == model_name]
        x = model_data['data_percentage']
        y = model_data['mean']
        yerr = model_data['std']

        plt.errorbar(x, y, yerr=yerr, label=model_name, marker='o', capsize=4)

    plt.title(f"Comparaison moyenne par modèle – {metric.upper()}")
    plt.xlabel('% des données utilisées')
    plt.ylabel(metric.upper())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(root_dir, f"average_subjectsfinals_{metric}_comparison.svg")
    plt.savefig(output_path)
    plt.close()
    print(f"[✓] Fichier sauvegardé : {output_path}")
