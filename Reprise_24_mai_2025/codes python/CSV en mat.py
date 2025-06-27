import os
import pandas as pd
from scipy.io import savemat

def merge_csvs_to_mat(input_folder, output_mat_path):
    # Liste tous les fichiers CSV dans le dossier
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    # Fusionne tous les DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    # Sauvegarde au format .mat
    savemat(output_mat_path, {'data': merged_df.to_numpy(), 'columns': merged_df.columns.to_list()})

# Exemple d'utilisation
input_folder = r'chemin\vers\ton\dossier'  # Remplace par ton dossier
output_mat_path = r'chemin\vers\ton\output\fusion.mat'  # Remplace par le chemin de sortie
merge_csvs_to_mat(input_folder, output_mat_path)
