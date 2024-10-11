import matlab.engine
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

# Démarrer le chronomètre
start_time = time.time()

# Démarrer le moteur MATLAB
eng = matlab.engine.start_matlab()

# Script MATLAB pour les données d'entraînement
matlab_script_train = 'dynamics_estimator_hysteresis'
# Script MATLAB pour les données de test
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

# Conversion des données en type float32 pour l'entraînement avec LSTM
X_train_combined = X_train_combined.astype(np.float32)
X_test_combined = X_test_combined.astype(np.float32)

# Encodage des étiquettes avec LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(gait_phases_train)
y_test_encoded = label_encoder.transform(gait_phases_test)

# Standardisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# Reshape pour le LSTM [samples, timesteps, features]
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Construction du modèle LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Supposons qu'il y ait 4 phases de marche

# Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train_reshaped, y_train_encoded, epochs=20, batch_size=32, validation_split=0.2)

# Prédiction sur les données de test
y_pred = model.predict(X_test_reshaped)
y_pred = y_pred.argmax(axis=1)

# Inverser l'encodage pour obtenir les phases sous forme de chaînes de caractères
y_pred_phases = label_encoder.inverse_transform(y_pred)
y_test_phases = label_encoder.inverse_transform(y_test_encoded)

# Évaluation du modèle
accuracy = accuracy_score(y_test_phases, y_pred_phases)
print(f"Précision du modèle LSTM : {accuracy * 100:.2f}%")

# Compter les valeurs prédites et vraies des classes
y_pred_values_count = pd.Series(y_pred_phases).value_counts()
y_pred_values_count_percent = pd.Series(y_pred_phases).value_counts(normalize=True)
y_test_values_count = pd.Series(y_test_phases).value_counts()
y_test_values_count_percent = pd.Series(y_test_phases).value_counts(normalize=True)

print("Nombre de valeurs prédites :")
print(y_pred_values_count)
print(y_pred_values_count_percent)
print("Nombre de valeurs vraies :")
print(y_test_values_count)
print(y_test_values_count_percent)

# Calcul du temps écoulé
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Temps d'exécution: {elapsed_time} secondes")

# Matrice de confusion
cm = confusion_matrix(y_test_phases, y_pred_phases)
class_names = label_encoder.classes_

# Visualisation de la matrice de confusion
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Matrice de confusion")
plt.ylabel('Étiquette réelle')
plt.xlabel('Étiquette prédite')
plt.show()