import matlab.engine
import time
import numpy as np
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Démarrer le chronomètre
start_time = time.time()

# Démarrer le moteur MATLAB
eng = matlab.engine.start_matlab()

# Script pour les données d'entraînement
matlab_script_train = 'dynamics_estimator_hysteresis'
# Script pour les données de test
matlab_script_test = 'dynamics_estimator_hysteresis_test'

# Extraction des données d'entraînement via MATLAB
eng.eval(matlab_script_train, nargout=0)
force_data_train = np.array(eng.workspace['force_data']).reshape(-1, 1)
gait_vector_train = np.array(eng.workspace['gait_vector']).flatten()
gait_phases_train = np.array(eng.workspace['gait_phases']).flatten()
ankle_angles_filt_train = np.array(eng.workspace['ankle_angles_filt']).flatten()

# Extraction des données de test via MATLAB
eng.eval(matlab_script_test, nargout=0)
force_data_test = np.array(eng.workspace['force_data']).reshape(-1, 1)
gait_vector_test = np.array(eng.workspace['gait_vector']).flatten()
gait_phases_test = np.array(eng.workspace['gait_phases']).flatten()
ankle_angles_filt_test = np.array(eng.workspace['ankle_angles_filt_test']).flatten()

# Arrêter le moteur MATLAB
eng.quit()

# Interpolation des angles pour les données d'entraînement (60 Hz vers 100 Hz)
t_60Hz_train = np.linspace(0, len(ankle_angles_filt_train) / 60, len(ankle_angles_filt_train))
t_100Hz_train = np.linspace(0, len(ankle_angles_filt_train) / 60, int(len(ankle_angles_filt_train) * 100 / 60))
interp_a_train = interp1d(t_60Hz_train, ankle_angles_filt_train, kind='linear')
ankle_angles_filt_train_100Hz = interp_a_train(t_100Hz_train)

# Interpolation des angles pour les données de test (60 Hz vers 100 Hz)
t_60Hz_test = np.linspace(0, len(ankle_angles_filt_test) / 60, len(ankle_angles_filt_test))
t_100Hz_test = np.linspace(0, len(ankle_angles_filt_test) / 60, int(len(ankle_angles_filt_test) * 100 / 60))
interp_a_test = interp1d(t_60Hz_test, ankle_angles_filt_test, kind='linear')
ankle_angles_filt_test_100Hz = interp_a_test(t_100Hz_test)

# Troncature des longueurs minimales
min_length_train = min(len(force_data_train), len(gait_phases_train), len(gait_vector_train), len(ankle_angles_filt_train_100Hz))
min_length_test = min(len(force_data_test), len(gait_phases_test), len(gait_vector_test), len(ankle_angles_filt_test_100Hz))

force_data_train = force_data_train[:min_length_train]
gait_phases_train = gait_phases_train[:min_length_train]
gait_vector_train = gait_vector_train[:min_length_train]
ankle_angles_filt_train_100Hz = ankle_angles_filt_train_100Hz[:min_length_train]

force_data_test = force_data_test[:min_length_test]
gait_phases_test = gait_phases_test[:min_length_test]
gait_vector_test = gait_vector_test[:min_length_test]
ankle_angles_filt_test_100Hz = ankle_angles_filt_test_100Hz[:min_length_test]

# Calcul de la dérivée des angles filtrés (entraînement et test)
ankle_angles_filt_train_100Hz_derivative = np.diff(ankle_angles_filt_train_100Hz)
ankle_angles_filt_test_100Hz_derivative = np.diff(ankle_angles_filt_test_100Hz)

# Combinaison des caractéristiques (force, vector et dérivées) pour l'entraînement
X_train_combined = np.hstack((force_data_train[:-1], gait_vector_train[:-1].reshape(-1, 1)))
X_train_combined = np.hstack((X_train_combined, np.diff(force_data_train, axis=0)))  # Dérivée de la force
X_train_combined = np.hstack((X_train_combined, ankle_angles_filt_train_100Hz[:-1].reshape(-1, 1)))
X_train_combined = np.hstack((X_train_combined, ankle_angles_filt_train_100Hz_derivative.reshape(-1, 1)))
gait_phases_train = gait_phases_train[:-2]
X_train_combined = X_train_combined[:-1]

# Combinaison des caractéristiques pour le test
X_test_combined = np.hstack((force_data_test[:-1], gait_vector_test[:-1].reshape(-1, 1)))
X_test_combined = np.hstack((X_test_combined, np.diff(force_data_test, axis=0)))
X_test_combined = np.hstack((X_test_combined, ankle_angles_filt_test_100Hz[:-1].reshape(-1, 1)))
X_test_combined = np.hstack((X_test_combined, ankle_angles_filt_test_100Hz_derivative.reshape(-1, 1)))
gait_phases_test = gait_phases_test[:-2]
X_test_combined = X_test_combined[:-1]

# Initialisation du modèle LightGBM
model = lgb.LGBMClassifier(n_estimators=100)

# Entraînement du modèle avec les données d'entraînement
model.fit(X_train_combined, gait_phases_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test_combined)

# Analyse des résultats
y_pred_values_count = pd.Series(y_pred).value_counts()
y_pred_values_count_percent = pd.Series(y_pred).value_counts(normalize=True)
y_test_values_count = pd.Series(gait_phases_test).value_counts()
y_test_values_count_percent = pd.Series(gait_phases_test).value_counts(normalize=True)

# Évaluation du modèle
accuracy = accuracy_score(gait_phases_test, y_pred)
print(f"LightGBM Model Accuracy: {accuracy * 100:.2f}%")
print(y_pred_values_count)
print(y_pred_values_count_percent)
print(y_test_values_count)
print(y_test_values_count_percent)

# Calcul du temps écoulé
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Temps d'exécution: {elapsed_time} secondes")

# Matrice de confusion
cm = confusion_matrix(gait_phases_test, y_pred)

class_names = ['FF/MST', 'HO', 'HS', 'MSW']

# Visualisation de la matrice de confusion
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion matrix")
plt.ylabel('Real classes')
plt.xlabel('Predicted classes')
plt.show()

# Importer les bibliothèques nécessaires
import matplotlib.pyplot as plt
import numpy as np

# Tracer les dérivées des angles interpolés pour les données d'entraînement
plt.figure(figsize=(12, 6))
plt.plot(ankle_angles_filt_train_100Hz_derivative, label="Angle derivative", color='blue')
plt.title("Derivatives of interpolated angles")
plt.xlabel("Samples")
plt.ylabel("Derivative")
plt.legend()
plt.grid()
plt.show()

# Tracer les dérivées des forces interpolées pour les données d'entraînement
force_derivative_train = np.diff(force_data_train, axis=0).flatten()
plt.figure(figsize=(12, 6))
plt.plot(force_derivative_train, label="Derivative of forces", color='green')
plt.title("Interpolated force derivatives")
plt.xlabel("Samples")
plt.ylabel("Derivative")
plt.legend()
plt.grid()
plt.show()