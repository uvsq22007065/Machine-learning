import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import time
import pandas as pd

# Démarrer le chronomètre
start_time = time.time()

# Chargement des données d'entraînement
train_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv"
data_train = pd.read_csv(train_file_path)
force_data_train = data_train['Force'].values
force_Derivative_data_train = data_train['Force_Derivative'].values
gait_vector_train = data_train['Gait_Progress'].values
ankle_angles_filt_train = data_train['Angle'].values
ankle_Derivative_angles_filt_train = data_train['Angle_Derivative'].values

# Préparation des caractéristiques et normalisation
def prepare_features(X, X_Derivative, a, a_derivative):
    return np.hstack((X.reshape(-1, 1), X_Derivative.reshape(-1, 1), a.reshape(-1, 1), a_derivative.reshape(-1, 1)))

X_train_combined = prepare_features(force_data_train, force_Derivative_data_train, ankle_angles_filt_train, ankle_Derivative_angles_filt_train)
y_train = gait_vector_train.flatten()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)

# Préparation des séquences
seq_length = 130
def create_sequences(data, labels, seq_length):
    sequences, label_sequences = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
        label_sequences.append(labels[i + seq_length - 1])
    return np.array(sequences), np.array(label_sequences)

X_seq_train, y_seq_train = create_sequences(X_train_scaled, y_train, seq_length)

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

# Charger les données de test
test_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv"
data_test = pd.read_csv(test_file_path)
force_data_test = data_test['Force'].values
force_Derivative_data_test = data_test['Force_Derivative'].values
gait_vector_test = data_test['Gait_Progress'].values
ankle_angles_filt_test = data_test['Angle'].values
ankle_Derivative_angles_filt_test = data_test['Angle_Derivative'].values

X_test_combined = prepare_features(force_data_test, force_Derivative_data_test, ankle_angles_filt_test, ankle_Derivative_angles_filt_test)
X_test_scaled = scaler.transform(X_test_combined)
y_test = gait_vector_test.flatten()

# Mémoire tampon pour prédictions en temps réel
buffer = []
y_pred_real_time = []
buf_length = 130

from collections import deque

# Mémoire tampon optimisée
buffer = deque(maxlen=buf_length)
y_pred_real_time = []

start_time2 = time.perf_counter()

for i in range(len(X_test_scaled)):
    buffer.append(X_test_scaled[i])
    if len(buffer) == buf_length:
        buffer_array = np.expand_dims(np.array(buffer), axis=0)
        pred = model.predict(buffer_array, verbose=0).flatten()[0]
        y_pred_real_time.append(pred)

end_time = time.perf_counter()

# Calculer le temps total et le temps moyen par itération
total_time = end_time - start_time2
average_time_per_iteration = total_time / len(X_test_scaled)

print(f"Temps total: {total_time:.4f} secondes")
print(f"Temps moyen par itération: {average_time_per_iteration:.4f} secondes")

# Calcul des erreurs et affichage des résultats
mse = mean_squared_error(y_test[buf_length-1:], y_pred_real_time)
mae = mean_absolute_error(y_test[buf_length-1:], y_pred_real_time)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Graphique de comparaison
plt.figure()
plt.plot(y_test[buf_length-1:], label="True gait progress")
plt.plot(y_pred_real_time, label="Real-time Prediction")
plt.xlabel("Samples")
plt.ylabel("Progression (%)")
plt.title("Comparaison gait progress")
plt.legend()
plt.show()
