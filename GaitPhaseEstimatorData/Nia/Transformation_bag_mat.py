import bagpy
from bagpy import bagreader
import pandas as pd
import scipy.io as sio
import numpy as np
import os

def bag_to_csv(bag_path):
    try:
        # Vérifier si le dossier CSV existe déjà
        output_dir = os.path.dirname(bag_path)
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]
        csv_dir = os.path.join(output_dir, f"{bag_name}_csv")
        
        if os.path.exists(csv_dir):
            print(f"Dossier CSV déjà existant pour : {bag_path}, conversion ignorée")
            return True
            
        os.makedirs(csv_dir, exist_ok=True)
        
        b = bagreader(bag_path)
        topics = b.topics
        
        for topic in topics:
            df_path = b.message_by_topic(topic)
            if df_path:
                # Nettoyer le nom du topic pour créer un nom de fichier valide
                topic_name = topic.replace('/', '_').strip('_')
                csv_path = os.path.join(csv_dir, f"{topic_name}.csv")
                
                # Lire et sauvegarder directement en CSV
                df = pd.read_csv(df_path)
                df.to_csv(csv_path, index=False)
                
        print(f"Conversion terminée pour : {bag_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la conversion de {bag_path}: {str(e)}")
        return False

def process_directory(root_dir):
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.bag'):
                bag_path = os.path.join(root, file)
                csv_dir = os.path.join(os.path.dirname(bag_path), 
                                     f"{os.path.splitext(file)[0]}_csv")
                
                if os.path.exists(csv_dir):
                    print(f"Fichier déjà traité : {bag_path}")
                    skipped_count += 1
                    continue
                    
                if bag_to_csv(bag_path):
                    success_count += 1
                else:
                    error_count += 1
    
    print(f"\nConversion terminée !")
    print(f"Fichiers convertis avec succès : {success_count}")
    print(f"Fichiers ignorés (déjà convertis) : {skipped_count}")
    print(f"Fichiers avec erreurs : {error_count}")

# Chemin du dossier racine à traiter
root_directory = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/Reprise_24_mai_2025"
process_directory(root_directory)
