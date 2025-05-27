import pandas as pd
import os

# Définir les chemins des dossiers
source_folder = "Reprise_24_mai_2025\\train_data_labeled"
destination_folder = "Reprise_24_mai_2025\\train_data_labeled_csv"

# Créer le dossier de destination s'il n'existe pas
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Parcourir tous les fichiers .xlsx dans le dossier source
for filename in os.listdir(source_folder):
    if filename.endswith('.xlsx'):
        # Lire le fichier Excel
        excel_file = pd.read_excel(os.path.join(source_folder, filename))
        
        # Créer le nouveau nom de fichier .csv
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        
        # Sauvegarder en CSV
        excel_file.to_csv(os.path.join(destination_folder, csv_filename), 
                         index=False,
                         encoding='utf-8')

print("Conversion terminée !")
