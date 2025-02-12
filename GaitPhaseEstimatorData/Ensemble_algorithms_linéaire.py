import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor

# Démarrer le chronomètre
start_time = time.time()

# Définir les chemins des fichiers CSV
train_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv"
test_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv"

# Charger les données
data_columns = ['Force', 'Force_Derivative', 'Gait_Progress', 'Phase', 'Angle', 'Angle_Derivative']
train_data = pd.read_csv(train_file_path)[data_columns]
test_data = pd.read_csv(test_file_path)[data_columns]

# Créer un mappage pour convertir les phases en chiffres
phase_mapping = {
    'stance_phase': 0,
    'swing_phase': 1,
    'double_stance': 2  # Ajoutez d'autres phases si nécessaire
}

# Appliquer le mappage à la colonne concernée
train_data['Phase'] = train_data['Phase'].map(phase_mapping)
test_data['Phase'] = test_data['Phase'].map(phase_mapping)

# Préparer les caractéristiques et les labels
X_train = train_data.drop(columns=['Gait_Progress']).values
y_train = train_data['Gait_Progress'].values
X_test = test_data.drop(columns=['Gait_Progress']).values
y_test = test_data['Gait_Progress'].values

# Standardiser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir les modèles
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
voting_model = VotingRegressor([('gb', model_gb), ('rf', model_rf)])

# Entraîner les modèles
model_gb.fit(X_train_scaled, y_train)
model_rf.fit(X_train_scaled, y_train)
voting_model.fit(X_train_scaled, y_train)

# Prédire les résultats
y_pred_gb = model_gb.predict(X_test_scaled)
y_pred_rf = model_rf.predict(X_test_scaled)
y_pred_voting = voting_model.predict(X_test_scaled)

# Calcul des erreurs pour chaque modèle
mse_gb = mean_squared_error(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_voting = mean_squared_error(y_test, y_pred_voting)
mae_voting = mean_absolute_error(y_test, y_pred_voting)

print(f"Gradient Boosting - MSE: {mse_gb:.2f}, MAE: {mae_gb:.2f}")
print(f"Random Forest - MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}")
print(f"Voting Regressor - MSE: {mse_voting:.2f}, MAE: {mae_voting:.2f}")

# Calculer le temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Tracer les résultats
plt.figure()
plt.plot(y_test, label="True gait progress", linestyle='--')
#plt.plot(y_pred_gb, label="Gradient Boosting")
#plt.plot(y_pred_rf, label="Random Forest")
plt.plot(y_pred_voting, label="Voting Regressor")
plt.xlabel("Samples")
plt.ylabel("Progress (%)")
plt.title("Model Comparison")
plt.legend()
plt.show()