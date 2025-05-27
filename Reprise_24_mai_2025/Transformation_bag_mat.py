import bagpy
from bagpy import bagreader
import pandas as pd
import scipy.io as sio
import numpy as np

# Chemin du fichier .bag en entrée et du .mat en sortie
bag_file = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/Nia/nia_walking3kmph.bag"
mat_file = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/Nia/nia_walking3kmph.mat"

def bag_to_mat(bag_path, mat_path):
    b = bagreader(bag_path)
    topics = b.topics  # Correction ici
    data_dict = {}
    
    for topic in topics:
        df_path = b.message_by_topic(topic)
        if df_path:
            df = pd.read_csv(df_path)  # Charger les données en DataFrame
            data_dict[topic] = df.to_dict(orient='list')  # Convertir en dictionnaire
    
    sio.savemat(mat_path, data_dict)
    print(f"Conversion terminée. Fichier enregistré sous : {mat_file}")

bag_to_mat(bag_file, mat_file)
