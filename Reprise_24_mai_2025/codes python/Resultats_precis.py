import os
import pandas as pd

# === Paramètres à modifier ici ===
subject_id = 1
percentages = [80]
models_to_ignore = ["RNNWS"]
root_dir = "Reprise_24_mai_2025//results"  # Chemin vers le dossier des résultats

# === Fonction principale ===
def get_model_performances(subject_id, percentages, root_dir, models_to_ignore=None):
    subject_path = os.path.join(root_dir, f"results_subject{subject_id}")
    results = []

    if not os.path.exists(subject_path):
        print(f"[❌] Le dossier pour le subject {subject_id} n'existe pas.")
        return pd.DataFrame()

    for folder in os.listdir(subject_path):
        model_path = os.path.join(subject_path, folder)
        if not os.path.isdir(model_path):
            continue

        csv_file = os.path.join(model_path, f"subject{subject_id}_overall_results.csv")
        if not os.path.exists(csv_file):
            continue

        try:
            df = pd.read_csv(csv_file)
            df.columns = [col.strip().lower() for col in df.columns]
        except Exception as e:
            print(f"[‼] Erreur de lecture dans {csv_file}: {e}")
            continue

        # Détection du modèle selon dossier
        if "Classical" in folder:
            pass  # Les noms de modèles sont dans la colonne 'model'
        else:
            model_name = folder.split('_')[0]
            df['model'] = model_name

        df['source_folder'] = folder
        df['subject'] = subject_id

        # Filtrer les pourcentages souhaités
        df_filtered = df[df['data_percentage'].isin(percentages)]
        results.append(df_filtered)

    if not results:
        print(f"[⚠] Aucun résultat trouvé pour le subject {subject_id}.")
        return pd.DataFrame()

    full_df = pd.concat(results, ignore_index=True)

    # Filtrer les modèles à ignorer
    if models_to_ignore:
        full_df = full_df[~full_df['model'].isin(models_to_ignore)]

    return full_df[['data_percentage', 'model','training_duration', 'rmse', 'mae', 'r2', 'final_train_loss', 'final_val_loss', 'total_epochs']]\
             .sort_values(by=['data_percentage', 'model'])\
             .reset_index(drop=True)

# === Exécution ===
df_perf = get_model_performances(subject_id, percentages, root_dir, models_to_ignore)

# === Affichage final ===
if not df_perf.empty:
    print("\n📊 Performances des modèles :\n")
    print(df_perf.to_string(index=False))
    # Sauvegarde dans un fichier CSV
    df_perf.to_csv(f"performances_subject{subject_id}.csv", sep=',', index=False)
    print(f"\n✅ Résultats sauvegardés dans performances_subject{subject_id}.csv")
else:
    print("Aucune donnée à afficher.")

