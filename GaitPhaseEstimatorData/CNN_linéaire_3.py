import matlab.engine
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import time
import pandas as pd

# Démarrer le chronomètre
start_time = time.time()

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

# Conversion des données en tableaux numpy
X_train = np.array(force_data_train).reshape(-1, 1)
y_train = np.array(gait_vector_train).flatten()
z_train = np.array(gait_phases_train).flatten()
a_train = np.array(ankle_angles_filt_train).flatten()

X_test = np.array(force_data_test).reshape(-1, 1)
y_test = np.array(gait_vector_test).flatten()
z_test = np.array(gait_phases_test).flatten()
a_test = np.array(ankle_angles_filt_test).flatten()

# Encodage des phases de marche
label_encoder = LabelEncoder()
z_train = label_encoder.fit_transform(z_train)
z_test = label_encoder.transform(z_test)

# Interpolation des données à 100 Hz
def interpolate_data(data, original_freq=60, target_freq=100):
    t_original = np.linspace(0, len(data) / original_freq, len(data))
    t_target = np.linspace(0, len(data) / original_freq, int(len(data) * target_freq / original_freq))
    interp_func = interp1d(t_original, data, kind='cubic')
    return interp_func(t_target)

a_train_100Hz = interpolate_data(a_train)
a_test_100Hz = interpolate_data(a_test)

# Calcul des dérivées
a_train_derivative = np.diff(a_train_100Hz, axis=0)
a_test_derivative = np.diff(a_test_100Hz, axis=0)

# Concaténation des caractéristiques
def prepare_features(X, a_100Hz, a_derivative):
    min_length = min(len(X), len(a_100Hz), len(a_derivative))
    X, a_100Hz, a_derivative = X[:min_length], a_100Hz[:min_length], a_derivative[:min_length]
    X_derivative = np.diff(X, axis=0)
    X = X[:-1]
    a_100Hz, a_derivative = a_100Hz[:-1], a_derivative[:-1]
    features = np.hstack((X, X_derivative, a_100Hz.reshape(-1, 1), a_derivative.reshape(-1, 1)))
    return features

X_combined_train = prepare_features(X_train, a_train_100Hz, a_train_derivative)
X_combined_test = prepare_features(X_test, a_test_100Hz, a_test_derivative)

# Standardisation des caractéristiques
scaler = StandardScaler()
X_combined_scaled_train = scaler.fit_transform(X_combined_train)
X_combined_scaled_test = scaler.transform(X_combined_test)

# Vérifier et égaliser les longueurs avant de créer des séquences
min_length_train = min(len(X_combined_scaled_train), len(y_train))
X_combined_scaled_train = X_combined_scaled_train[:min_length_train]
y_train = y_train[:min_length_train]

min_length_test = min(len(X_combined_scaled_test), len(y_test))
X_combined_scaled_test = X_combined_scaled_test[:min_length_test]
y_test = y_test[:min_length_test]

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

# Gestion des données en batchs avec tf.data.Dataset
train_data = tf.data.Dataset.from_tensor_slices((X_seq_train, y_seq_train)).batch(32)
test_data = tf.data.Dataset.from_tensor_slices((X_seq_test, y_seq_test)).batch(32)

# Modèle CNN amélioré avec Bidirectional LSTM
model = Sequential([
    Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(50, return_sequences=False)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Callback pour l'early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Ajustement du modèle
model.fit(train_data, epochs=20, validation_data=test_data, callbacks=[early_stopping])

# Prédictions
y_pred = model.predict(X_seq_test).flatten()

# Calcul des erreurs
mse = mean_squared_error(y_seq_test, y_pred)
mae = mean_absolute_error(y_seq_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Visualisation des résultats
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

# Imprimer les informations pour train_data
for features_batch, labels_batch in train_data.take(1):  # On prend seulement le premier batch
    print("Train Data:")
    print("Type de features:", type(features_batch))
    print("Taille de features:", features_batch.shape)
    print("Matrice de features:\n", features_batch.numpy())
    
    print("Type de labels:", type(labels_batch))
    print("Taille de labels:", labels_batch.shape)
    print("Matrice de labels:\n", labels_batch.numpy())

# Imprimer les informations pour test_data
for features_batch, labels_batch in test_data.take(1):  # On prend seulement le premier batch
    print("\nTest Data:")
    print("Type de features:", type(features_batch))
    print("Taille de features:", features_batch.shape)
    print("Matrice de features:\n", features_batch.numpy())
    
    print("Type de labels:", type(labels_batch))
    print("Taille de labels:", labels_batch.shape)
    print("Matrice de labels:\n", labels_batch.numpy())