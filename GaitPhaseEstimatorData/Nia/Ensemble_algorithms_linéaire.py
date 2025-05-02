import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor

# Démarrer le chronomètre
start_time = time.time()
# Chemins des fichiers
import os
# Chemins des fichiers
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Fonction pour charger et préparer les données d'un dossier
def load_and_prepare_data(folder_path):
    # Chemins vers les fichiers nécessaires
    ankle_file = os.path.join(folder_path, "ankle_joint-angle.csv")
    vgrf_file = os.path.join(folder_path, "vGRF.csv")
    gait_progress_file = os.path.join(folder_path, "gait_progress.csv")
    
    # Vérifier que tous les fichiers existent
    if not all(os.path.exists(f) for f in [ankle_file, vgrf_file, gait_progress_file]):
        print(f"Certains fichiers sont manquants dans le dossier {folder_path}")
        return None
    
    # Lire les fichiers CSV
    ankle_data = pd.read_csv(ankle_file)
    vgrf_data = pd.read_csv(vgrf_file)
    gait_progress_data = pd.read_csv(gait_progress_file)
    
    # Renommer les colonnes pour clarté
    ankle_data = ankle_data.rename(columns={"data": "Angle"})
    vgrf_data = vgrf_data.rename(columns={"data": "Force"})
    
    # Vérifier que ankle_data et gait_progress_data ont le même nombre de lignes
    if len(ankle_data) != len(gait_progress_data):
        print(f"Les fichiers ankle_data et gait_progress_data n'ont pas le même nombre de lignes dans {folder_path}")
        return None
    
    # Créer un nouveau DataFrame avec toutes les données nécessaires
    merged_data = pd.DataFrame()
    merged_data['Time'] = ankle_data['Time']
    merged_data['Angle'] = ankle_data['Angle']
    merged_data['Gait_Progress'] = gait_progress_data['Gait_Progress']
    
    # Interpoler les données de force sur les points de temps de ankle_data
    from scipy.interpolate import interp1d
    
    # Créer une fonction d'interpolation pour la force
    force_interp = interp1d(vgrf_data['Time'], vgrf_data['Force'], 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Appliquer l'interpolation pour obtenir les valeurs de force aux mêmes temps que ankle_data
    merged_data['Force'] = force_interp(merged_data['Time'])
    
    # Calculer les dérivées
    merged_data['Angle_Derivative'] = np.gradient(merged_data['Angle'])
    merged_data['Force_Derivative'] = np.gradient(merged_data['Force'])
    
    # Supprimer les lignes avec des valeurs NaN
    merged_data = merged_data.dropna()
    
    print(f"Données préparées pour {folder_path}: {len(merged_data)} lignes")
    return merged_data

# Demander les chemins des dossiers d'entraînement et de test
train_folder = input("Entrez le chemin du dossier contenant les données d'entraînement : ")
test_folder = input("Entrez le chemin du dossier contenant les données de test : ")

# Charger et préparer les données d'entraînement
train_data = load_and_prepare_data(train_folder)

# Charger et préparer les données de test
test_data = load_and_prepare_data(test_folder)

# Vérifier que les données ont été chargées correctement
if train_data is None or test_data is None:
    print("Erreur lors du chargement des données. Veuillez vérifier les dossiers et fichiers.")
    # Définir des variables vides pour éviter les erreurs
    features_train = np.array([[0, 0, 0, 0]])
    labels_train = np.array([0])
    features_test = np.array([[0, 0, 0, 0]])
    labels_test = np.array([0])
else:
    # Extraire features et labels
    features_train = train_data[['Force', 'Force_Derivative', 'Angle', 'Angle_Derivative']].values
    labels_train = train_data['Gait_Progress'].values
    features_test = test_data[['Force', 'Force_Derivative', 'Angle', 'Angle_Derivative']].values
    labels_test = test_data['Gait_Progress'].values
    
    print(f"Données d'entraînement: {len(features_train)} échantillons")
    print(f"Données de test: {len(features_test)} échantillons")

# Standardisation des caractéristiques
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Préparer les caractéristiques et les labels
features_train = train_data.drop(columns=['Gait_Progress']).values
labels_train = train_data['Gait_Progress'].values
features_test = test_data.drop(columns=['Gait_Progress']).values
labels_test = test_data['Gait_Progress'].values

# Standardiser les données
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

print("Préparation des données terminée avec succès.")

# Définir les modèles
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
voting_model = VotingRegressor([('gb', model_gb), ('rf', model_rf)])

# Entraîner les modèles
model_gb.fit(features_train_scaled, labels_train)
model_rf.fit(features_train_scaled, labels_train)
voting_model.fit(features_train_scaled, labels_train)

# Prédire les résultats
labels_pred_gb = model_gb.predict(features_test_scaled)
labels_pred_rf = model_rf.predict(features_test_scaled)
labels_pred_voting = voting_model.predict(features_test_scaled)

# Calcul des erreurs pour chaque modèle
mse_gb = mean_squared_error(labels_test, labels_pred_gb)
mae_gb = mean_absolute_error(labels_test, labels_pred_gb)
mse_rf = mean_squared_error(labels_test, labels_pred_rf)
mae_rf = mean_absolute_error(labels_test, labels_pred_rf)
mse_voting = mean_squared_error(labels_test, labels_pred_voting)
mae_voting = mean_absolute_error(labels_test, labels_pred_voting)

print(f"Gradient Boosting - MSE: {mse_gb:.2f}, MAE: {mae_gb:.2f}")
print(f"Random Forest - MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}")
print(f"Voting Regressor - MSE: {mse_voting:.2f}, MAE: {mae_voting:.2f}")

# Calculer le temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Tracer les résultats
plt.figure()
plt.plot(labels_test, label="True gait progress", linestyle='--')
plt.plot(labels_pred_gb, label="Gradient Boosting")
plt.plot(labels_pred_rf, label="Random Forest")
plt.plot(labels_pred_voting, label="Voting Regressor")
plt.xlabel("Samples")
plt.ylabel("Progress (%)")
plt.title("Model Comparison")
plt.legend()
plt.show()