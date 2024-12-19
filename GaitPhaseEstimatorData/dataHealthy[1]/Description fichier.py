import scipy.io
import numpy as np

def describe_mat_file(file_path):
    try:
        # Charger le fichier .mat
        mat_data = scipy.io.loadmat(file_path)
        
        print(f"Structure du fichier .mat: {file_path}\n")
        
        for key, value in mat_data.items():
            # Ignorer les métadonnées de .mat
            if key.startswith("__") and key.endswith("__"):
                continue

            print(f"Catégorie : {key}")

            # Vérifier si c'est un tableau
            if isinstance(value, np.ndarray):
                print(f"  Longueur : {value.size}")

                # Vérifier si la catégorie contient des sous-catégories
                if value.dtype.names:
                    print(f"  Sous-catégories :")
                    for sub_key in value.dtype.names:
                        sub_value = value[sub_key][0, 0] if value[sub_key].size > 0 else None
                        sub_length = sub_value.size if sub_value is not None else 0
                        print(f"    - {sub_key}: Longueur {sub_length}")
                else:
                    print("  Pas de sous-catégories.")
            else:
                print(f"  Type non supporté : {type(value)}")

            print("\n")

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")

# Exemple d'utilisation
file_path = "C:\\Users\\Grégoire\\OneDrive\\Bureau\\EPF\\BRL\\Machine learning\\GaitPhaseEstimatorData\\dataHealthy[1]\\Subject1_left.mat"  # Remplacez par le chemin vers votre fichier .mat
describe_mat_file(file_path)
