import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import time
import pandas as pd
import os

# Démarrer le chronomètre
start_time = time.time()

# Définir les chemins des fichiers CSV pour les données d'entraînement et de test
train_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv"
test_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv"

# Lire les fichiers CSV
force_data_train = pd.read_csv(train_file_path)['Force'].values
force_Derivative_data_train = pd.read_csv(train_file_path)['Force_Derivative'].values
gait_vector_train = pd.read_csv(train_file_path)['Gait_Progress'].values
gait_phases_train = pd.read_csv(train_file_path)['Phase'].values
ankle_angles_filt_train = pd.read_csv(train_file_path)['Angle'].values
ankle_Derivative_angles_filt_train = pd.read_csv(train_file_path)['Angle_Derivative'].values

force_data_test = pd.read_csv(test_file_path)['Force'].values
force_Derivative_data_test = pd.read_csv(test_file_path)['Force_Derivative'].values
gait_vector_test = pd.read_csv(test_file_path)['Gait_Progress'].values
gait_phases_test = pd.read_csv(test_file_path)['Phase'].values
ankle_angles_filt_test = pd.read_csv(test_file_path)['Angle'].values
ankle_Derivative_angles_filt_test = pd.read_csv(test_file_path)['Angle_Derivative'].values

# Vous pouvez maintenant utiliser ces variables dans votre code
print("Données d'entraînement et de test chargées avec succès.")

# Conversion des données en tableaux numpy
X_train = np.array(force_data_train).reshape(-1, 1)
y_train = np.array(gait_vector_train).flatten()
X_train_Derivative = np.array(force_Derivative_data_train).flatten()
a_train = np.array(ankle_angles_filt_train).flatten()
a_train_Derivative = np.array(ankle_Derivative_angles_filt_train).flatten()

X_test = np.array(force_data_test).reshape(-1, 1)
y_test = np.array(gait_vector_test).flatten()
X_test_Derivative = np.array(force_Derivative_data_test).flatten()
a_test = np.array(ankle_angles_filt_test).flatten()
a_test_Derivative = np.array(ankle_Derivative_angles_filt_test).flatten()

# Concaténation des caractéristiques
def prepare_features(X, X_Derivative, a, a_derivative):
    features = np.hstack((X.reshape(-1, 1), 
                          X_Derivative.reshape(-1, 1), 
                          a.reshape(-1, 1), 
                          a_derivative.reshape(-1, 1)))
    return features

X_combined_train = prepare_features(X_train, X_train_Derivative, a_train, a_train_Derivative)
X_combined_test = prepare_features(X_test, X_test_Derivative, a_test, a_test_Derivative)

# Standardisation des caractéristiques
scaler = StandardScaler()
X_combined_scaled_train = scaler.fit_transform(X_combined_train)
X_combined_scaled_test = scaler.transform(X_combined_test)

# Vérifier et égaliser les longueurs avant de créer des séquences
min_length_train = min(len(X_combined_scaled_train), len(y_train))
X_combined_scaled_train = X_combined_scaled_train[:min_length_train]
y_train = y_train[:min_length_train]

min_length_test = min(len(X_combined_scaled_test), len(y_test))
X_combined_scaled_test = X_combined_scaled_test[-min_length_test:]
y_test = y_test[-min_length_test:]

# Création des séquences pour le Conv1D et LSTM
def create_sequences(data, labels, seq_length):
    sequences, label_sequences = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
        label_sequences.append(labels[i + seq_length - 1])
    return np.array(sequences), np.array(label_sequences)

seq_length = 130
X_seq_train, y_seq_train = create_sequences(X_combined_scaled_train, y_train, seq_length)
X_seq_test, y_seq_test = create_sequences(X_combined_scaled_test, y_test, seq_length)

# Modèle CNN amélioré avec LSTM
model = Sequential([
    Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(50, return_sequences=False),  # Ajout d'une couche LSTM
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compiler le modèle avec un learning rate plus bas et une fonction de perte adaptée
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callback pour l'early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Ajustement du modèle avec plus d'epochs et le callback d'early stopping
model.fit(
    X_seq_train, y_seq_train, 
    epochs=10,  # Augmentation du nombre d'epochs
    batch_size=32,
    callbacks=[early_stopping]  # Ajout de l'early stopping
)

# Entraînement du modèle avec validation croisée
model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, callbacks=[early_stopping], verbose=1)

# Prédictions
y_pred = model.predict(X_seq_test).flatten()

# Calculer le temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Calcul des erreurs
mse = mean_squared_error(y_seq_test, y_pred)
mae = mean_absolute_error(y_seq_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Tracé de la progression réelle vs prédite
plt.figure()
plt.plot(y_seq_test, label="True gait progress")
plt.plot(y_pred, label="Prediction")
plt.xlabel("Samples")
plt.ylabel("Progression (%)")
plt.title("Comparaison gait progress")
plt.legend()
plt.show()

# Scatter plot pour l'analyse des erreurs de prédiction
plt.figure()
plt.scatter(y_seq_test, y_pred, alpha=0.5)
plt.xlabel("Vrai Gait Progress")
plt.ylabel("Prédiction")
plt.title("Vrai vs Prédiction Gait Progress")
plt.grid(True)
plt.show()
