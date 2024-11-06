import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import time
import pandas as pd

# Démarrer le chronomètre
start_time = time.time()

# Charger les données d'entraînement et de test
train_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv"
test_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv"

force_data_train = pd.read_csv(train_file_path)['Force'].values
force_Derivative_data_train = pd.read_csv(train_file_path)['Force_Derivative'].values
gait_vector_train = pd.read_csv(train_file_path)['Gait_Progress'].values
ankle_angles_filt_train = pd.read_csv(train_file_path)['Angle'].values
ankle_Derivative_angles_filt_train = pd.read_csv(train_file_path)['Angle_Derivative'].values

force_data_test = pd.read_csv(test_file_path)['Force'].values
force_Derivative_data_test = pd.read_csv(test_file_path)['Force_Derivative'].values
gait_vector_test = pd.read_csv(test_file_path)['Gait_Progress'].values
ankle_angles_filt_test = pd.read_csv(test_file_path)['Angle'].values
ankle_Derivative_angles_filt_test = pd.read_csv(test_file_path)['Angle_Derivative'].values

# Préparer les données
def prepare_features(X, X_Derivative, a, a_derivative):
    return np.hstack((X.reshape(-1, 1), X_Derivative.reshape(-1, 1), a.reshape(-1, 1), a_derivative.reshape(-1, 1)))

X_combined_train = prepare_features(force_data_train, force_Derivative_data_train, ankle_angles_filt_train, ankle_Derivative_angles_filt_train)
X_combined_test = prepare_features(force_data_test, force_Derivative_data_test, ankle_angles_filt_test, ankle_Derivative_angles_filt_test)

scaler = StandardScaler()
X_combined_scaled_train = scaler.fit_transform(X_combined_train)
X_combined_scaled_test = scaler.transform(X_combined_test)

# Limiter les données si nécessaire
min_length_train = min(len(X_combined_scaled_train), len(gait_vector_train))
X_combined_scaled_train = X_combined_scaled_train[:min_length_train]
y_train = gait_vector_train[:min_length_train]

min_length_test = min(len(X_combined_scaled_test), len(gait_vector_test))
X_combined_scaled_test = X_combined_scaled_test[:min_length_test]
y_test = gait_vector_test[:min_length_test]

# Création des séquences
def create_sequences(data, labels, seq_length):
    sequences, label_sequences = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
        label_sequences.append(labels[i + seq_length - 1])
    return np.array(sequences), np.array(label_sequences)

seq_length = 130
X_seq_train, y_seq_train = create_sequences(X_combined_scaled_train, y_train, seq_length)
X_seq_test, y_seq_test = create_sequences(X_combined_scaled_test, y_test, seq_length)

# Modèle NRAX
input_layer = Input(shape=(X_seq_train.shape[1], X_seq_train.shape[2]))

conv_block = Conv1D(128, kernel_size=3, activation='relu')(input_layer)
conv_block = MaxPooling1D(pool_size=2)(conv_block)
conv_block = Conv1D(64, kernel_size=3, activation='relu')(conv_block)
conv_block = MaxPooling1D(pool_size=2)(conv_block)
flatten_layer = Flatten()(conv_block)

lstm_block = LSTM(50)(input_layer)

combined = concatenate([flatten_layer, lstm_block])
dense_layer = Dense(32, activation='relu')(combined)
output_layer = Dense(1, activation='linear')(dense_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callback pour l'early stopping
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Entraîner le modèle
model.fit(X_seq_train, y_seq_train, epochs=15, batch_size=32, callbacks=[early_stopping], validation_data=(X_seq_test, y_seq_test))

# Prédictions
y_pred = model.predict(X_seq_test).flatten()

# Temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Calcul des erreurs
mse = mean_squared_error(y_seq_test, y_pred)
mae = mean_absolute_error(y_seq_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Tracé
plt.figure()
plt.plot(y_seq_test, label="True gait progress")
plt.plot(y_pred, label="Prediction")
plt.xlabel("Samples")
plt.ylabel("Progression (%)")
plt.title("Comparaison gait progress")
plt.legend()
plt.show()
