import matlab.engine
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
import time

# Démarrer le chronomètre
start_time = time.time()

import pandas as pd
# Définir les chemins des fichiers CSV pour les données d'entraînement et de test
train_file_path = 'C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv'
test_file_path = 'C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv'

# Lire les fichiers CSV
force_data_train = pd.read_csv(train_file_path)['Force'].values
gait_vector_train = pd.read_csv(train_file_path)['Gait_Progress'].values
gait_phases_train = pd.read_csv(train_file_path)['Phase'].values
ankle_angles_filt_train = pd.read_csv(train_file_path)['Angle'].values

force_data_test = pd.read_csv(test_file_path)['Force'].values
gait_vector_test = pd.read_csv(test_file_path)['Gait_Progress'].values
gait_phases_test = pd.read_csv(test_file_path)['Phase'].values
ankle_angles_filt_test = pd.read_csv(test_file_path)['Angle'].values

# Vous pouvez maintenant utiliser ces variables dans votre code
print("Données d'entraînement et de test chargées avec succès.")

# Assuming 'force_data', 'gait_phases', and 'gait_vector' are arrays from the MATLAB output
X_train = np.array(force_data_train).reshape(-1, 1)  # Force data as feature
y_train = np.array(gait_vector_train)  # Gait vector as labels (progression)
z_train = np.array(gait_phases_train)
a_train = np.array(ankle_angles_filt_train)

# Testing data (to predict)
X_test = np.array(force_data_test).reshape(-1, 1)
z_test = np.array(gait_phases_test)
y_test = np.array(gait_vector_test)
a_test = np.array(ankle_angles_filt_test)

# Flatten arrays if needed
y_train = y_train.flatten()
z_train = z_train.flatten()
a_train = a_train.flatten()
z_test = z_test.flatten()
a_test = a_test.flatten()

# Encode gait phases (y) into integers
label_encoder = LabelEncoder()
z_train = label_encoder.fit_transform(z_train)  # Transform gait phases to integers
z_test = label_encoder.fit_transform(z_test)  # Transform gait phases to integers

# Interpolation de la variable a_train (à 60 Hz) vers 100 Hz
t_60Hz_train = np.linspace(0, len(a_train) / 60, len(a_train))
t_100Hz_train = np.linspace(0, len(a_train) / 60, int(len(a_train) * 100 / 60))
interp_a_train = interp1d(t_60Hz_train, a_train, kind='linear')
a_train_100Hz = interp_a_train(t_100Hz_train)

# Interpolation de la variable a_test (à 60 Hz) vers 100 Hz
t_60Hz_test = np.linspace(0, len(a_test) / 60, len(a_test))
t_100Hz_test = np.linspace(0, len(a_test) / 60, int(len(a_test) * 100 / 60))
interp_a_test = interp1d(t_60Hz_test, a_test, kind='linear')
a_test_100Hz = interp_a_test(t_100Hz_test)

# Find the minimum length among X, y, z, and interpolated a_train
min_length_train = min(len(X_train), len(y_train), len(z_train), len(a_train_100Hz))

# Truncate X, y, z, and interpolated a_train to the minimum length
X_train = X_train[:min_length_train]
y_train = y_train[:min_length_train]
z_train = z_train[:min_length_train]
a_train_100Hz = a_train_100Hz[:min_length_train]

# Derivatives of interpolated a_train
a_train_derivate = np.diff(a_train_100Hz, axis=0)

# Combine features: X (force), z (gait vector), a (angles) and derivatives
X_combined_train = np.hstack((X_train, z_train.reshape(-1, 1)))
X_derivative_train = np.diff(X_train, axis=0)  # Calcul des dérivées de la force

X_combined_train = X_combined_train[:-1]
X_combined_train = np.hstack((X_combined_train, X_derivative_train))
X_combined_train = np.hstack((X_combined_train, a_train_100Hz.reshape(-1, 1)[:-1]))
X_combined_train = np.hstack((X_combined_train, a_train_derivate.reshape(-1, 1)))
y_train = y_train[:-2]

# Standardize the features
scaler = StandardScaler()
X_combined_scaled_train = scaler.fit_transform(X_combined_train)

# Fenêtrage des données pour Conv1D
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 10
X_combined_reshaped_train = create_sequences(X_combined_scaled_train, seq_length)
y_seq_train = y_train[seq_length-1:]

# Interpolation de la variable a_test et combinaison similaire aux données d'entraînement
min_length_test = min(len(X_test), len(z_test), len(a_test_100Hz))

X_test = X_test[:min_length_test]
z_test = z_test[:min_length_test]
a_test_100Hz = a_test_100Hz[:min_length_test]

a_test_derivate = np.diff(a_test_100Hz, axis=0)

X_combined_test = np.hstack((X_test, z_test.reshape(-1, 1)))
X_derivative_test = np.diff(X_test, axis=0)

X_combined_test = X_combined_test[:-1]
X_combined_test = np.hstack((X_combined_test, X_derivative_test))
X_combined_test = np.hstack((X_combined_test, a_test_100Hz.reshape(-1, 1)[:-1]))
X_combined_test = np.hstack((X_combined_test, a_test_derivate.reshape(-1, 1)))

# Standardize the features
X_combined_scaled_test = scaler.transform(X_combined_test)

# Créer des fenêtres pour les données de test
X_combined_reshaped_test = create_sequences(X_combined_scaled_test, seq_length)
y_seq_test = y_test[seq_length-1:]
X_combined_reshaped_train = X_combined_reshaped_train[:-1]

# Construire le modèle CNN
model = Sequential()
model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_combined_reshaped_train.shape[1], X_combined_reshaped_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Sortie continue pour la régression

# Compiler le modèle avec une fonction de perte adaptée
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Perte MSE pour la régression

# Ajustement du modèle
model.fit(X_combined_reshaped_train, y_seq_train, epochs=50, batch_size=32)

# Prédiction sur les données de test
y_pred = model.predict(X_combined_reshaped_test).flatten()
y_seq_test = y_seq_test.flatten()

# Find the minimum length among X, y, z, and interpolated a_train
min_length_final = min(len(y_seq_test), len(y_pred))

# Truncate X, y, z, and interpolated a_train to the minimum length
y_pred_first = y_pred[-min_length_final:]
y_seq_test_first = y_seq_test[-min_length_final:]

mse_first = mean_squared_error(y_seq_test_first, y_pred_first)
mae_first = mean_absolute_error(y_seq_test_first, y_pred_first)
print(f"Mean Squared Error: {mse_first:.2f}")
print(f"Mean Absolute Error: {mae_first:.2f}")

y_pred_second = y_pred[:min_length_final]
y_seq_test_second = y_seq_test[:min_length_final]

mse_second = mean_squared_error(y_seq_test_second, y_pred_second)
mae_second = mean_absolute_error(y_seq_test_second, y_pred_second)
print(f"Mean Squared Error 2: {mse_second:.2f}")
print(f"Mean Absolute Error 2: {mae_second:.2f}")

# Arrêter le chronomètre
end_time = time.time()

# Calculer le temps écoulé
elapsed_time = end_time - start_time
print(f"Temps d'exécution: {elapsed_time} secondes")

# Tracé de la progression réelle vs prédite
plt.figure()
plt.plot(y_seq_test_first, label="True gait progress")
plt.plot(y_pred_first, label="Prediction")
plt.xlabel("Samples")
plt.ylabel("Progression (%)")
plt.title("Comparaison gait progress")
plt.legend()
plt.show()
