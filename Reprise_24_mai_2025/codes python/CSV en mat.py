import os
import pandas as pd
from scipy.io import savemat
import shutil
import time
import threading
import re  # Ajout pour extraction du pct

def show_mat_progress(estimated_size, done_flag, start_time):
    speed = 100 * 1024 * 1024  # 100 Mo/s arbitraire
    estimated_total_time = estimated_size / speed
    while not done_flag[0]:
        elapsed = time.time() - start_time[0]
        remaining = estimated_total_time - elapsed
        if remaining < 0:
            remaining = 0
        print(
            f"Transformation en .mat... Taille estimée : {estimated_size//(1024**2)} Mo | "
            f"Vitesse estimée : 100 Mo/s | "
            f"Temps écoulé : {int(elapsed)//60} min {int(elapsed)%60} s | "
            f"Temps restant estimé : {int(remaining)//60} min {int(remaining)%60} s",
            end='\r'
        )
        time.sleep(1)
    elapsed = time.time() - start_time[0]
    print(f"Transformation en .mat terminée en {int(elapsed)//60} min {int(elapsed)%60} s.{' '*30}")

def merge_csvs_to_mat(input_folder, output_mat_path):
    # Parcourt récursivement tous les sous-dossiers pour trouver les fichiers CSV
    model_data = {}
    model_sources = {}
    overall_results = {}  # Pour stocker les overall_results
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                model_folder = os.path.basename(root)
                # Détection spéciale pour les overall_results
                if file.endswith('_overall_results.csv'):
                    # Utilise le nom du modèle ou du dossier parent pour la clé
                    overall_key = f"{model_folder}_overall_results"
                    df = pd.read_csv(file_path)
                    # Conversion pour compatibilité MATLAB : tout en float ou string
                    try:
                        arr = df.to_numpy(dtype=float)
                    except Exception:
                        arr = df.astype(str).to_numpy()
                    overall_results[overall_key] = {
                        'data': arr,
                        'columns': list(df.columns),
                        'index': list(df.index)
                    }
                    continue
                # Pour Classical, différencie les modèles
                if model_folder.lower() == "classical":
                    # Cherche le nom du modèle dans le nom du fichier
                    if "Gradient_Boosting" in file:
                        model_name = "Classical_Gradient_Boosting"
                    elif "Random_Forest" in file:
                        model_name = "Classical_Random_Forest"
                    elif "Voting_Regressor" in file:
                        model_name = "Classical_Voting_Regressor"
                    else:
                        model_name = "Classical"
                else:
                    model_name = model_folder.split('_')[0] if '_' in model_folder else model_folder
                df = pd.read_csv(file_path)
                col_signature = tuple(df.columns)
                if model_name not in model_data:
                    model_data[model_name] = {}
                    model_sources[model_name] = {}
                if col_signature not in model_data[model_name]:
                    model_data[model_name][col_signature] = []
                    model_sources[model_name][col_signature] = []
                model_data[model_name][col_signature].append(df)
                model_sources[model_name][col_signature].append(file_path)
    if not model_data and not overall_results:
        raise ValueError("Aucun fichier CSV trouvé dans le dossier et ses sous-dossiers.")

    # Affichage de la progression de lecture
    total_files = sum(len(lst) for m in model_data.values() for lst in m.values())
    idx = 0
    start_time_read = time.time()
    for model_name, type_dict in model_data.items():
        for col_signature, dfs in type_dict.items():
            for df in dfs:
                idx += 1
                elapsed = time.time() - start_time_read
                avg_time = elapsed / idx
                remaining = total_files - idx
                if remaining > 0:
                    est_left = avg_time * remaining
                    print(f"Lecture {idx}/{total_files} : {model_name} | Temps restant estimé : {int(est_left)//60} min {int(est_left)%60} s", end='\r')
                else:
                    print(f"Lecture {idx}/{total_files} terminée.                          ", end='\r')
    print()

    # Nouvelle structure compatible savemat (structure plate)
    mat_dict = {}
    sources_dict = {}
    for model_name in sorted(model_data.keys()):
        for col_signature, dfs in model_data[model_name].items():
            # On va parcourir chaque DataFrame et son fichier source pour extraire le pct du nom du fichier
            for df, src in zip(dfs, model_sources[model_name][col_signature]):
                # Extraction du pct dans le nom du fichier, ex: _pct20 ou -pct20 ou pct20
                match = re.search(r'[_\-]?pct(\d+)', os.path.basename(src), re.IGNORECASE)
                pct_str = f"pct_{match.group(1)}" if match else None
                # Construction du nom de la clé
                key_parts = [model_name] + list(col_signature)
                if pct_str:
                    key_parts.append(pct_str)
                type_name = "_".join(key_parts)
                if len(type_name) > 60:
                    type_name = type_name[:57] + "..."
                mat_dict[type_name] = {
                    'data': df.to_numpy(),
                    'columns': list(col_signature)
                }
                sources_dict[type_name] = [src]

    # Ajoute les overall_results dans le mat_dict
    for key, val in overall_results.items():
        mat_dict[key] = val

    # Ajoute une clé d'information générale
    mat_dict['__info__'] = {
        'description': (
            "Ce fichier .mat contient les données fusionnées de chaque modèle IA.\n"
            "Chaque clé correspond à un modèle+type (XXX_col1_col2...), contenant :\n"
            "- data : les données fusionnées (numpy array)\n"
            "- columns : noms des colonnes\n"
            "Les clés '*_overall_results' contiennent les métriques globales (R², MAE, etc).\n"
            "La clé '__info__' contient ce message.\n"
            "La clé '__sources__' contient la liste des fichiers sources utilisés."
        ),
        'models': sorted(list(model_data.keys()))
    }
    mat_dict['__sources__'] = sources_dict

    # Vérifie l'espace disque disponible
    output_dir = os.path.dirname(output_mat_path)
    total, used, free = shutil.disk_usage(output_dir)
    estimated_size = sum(df.memory_usage(deep=True).sum() for m in model_data.values() for lst in m.values() for df in lst) * 2
    if free < estimated_size:
        raise OSError(f"Espace disque insuffisant ({free // (1024**2)} Mo libres, {estimated_size // (1024**2)} Mo nécessaires)")

    # Affichage récapitulatif de la structure trouvée
    print("\nRésumé de la structure détectée :")
    for model_name in sorted(model_data.keys()):
        print(f"Modèle : {model_name}")
        for col_signature, dfs in model_data[model_name].items():
            sources = model_sources[model_name][col_signature]
            print(f"  Type : {col_signature} | CSV trouvés : {len(sources)}")
            for src, df in zip(sources, dfs):
                print(f"    - {src} | Lignes : {len(df)}")

    # Debug : afficher la taille estimée des données à sauvegarder
    print(f"\nTaille totale estimée des données à sauvegarder : {estimated_size // (1024**2)} Mo")
    print(f"Nombre de clés dans mat_dict : {len(mat_dict)}")

    # Sauvegarde au format .mat avec estimateur de temps en temps réel et estimation du temps restant
    print("\nDébut de la transformation en .mat...")
    start_time_mat = [time.time()]
    done_flag = [False]
    progress_thread = threading.Thread(target=show_mat_progress, args=(estimated_size, done_flag, start_time_mat))
    progress_thread.start()
    savemat(output_mat_path, mat_dict)
    done_flag[0] = True
    progress_thread.join()

# Exemple d'utilisation
input_folder = r'Reprise_24_mai_2025\\results\\results_subject1'  # Remplace par ton dossier principal contenant les sous-dossiers de modèles
output_mat_path = r'Reprise_24_mai_2025\\results\\fusion\\fusion_version_finale_finale_subject1.mat'  # Remplace par le chemin de sortie
merge_csvs_to_mat(input_folder, output_mat_path)
