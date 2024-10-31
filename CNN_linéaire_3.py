import matlab.engine
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, LSTM, Dropout, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import time

# Démarrer le chronomètre
start_time = time.time()

# Démarrer le moteur MATLAB
eng = matlab.engine.start_matlab()

# Définir les chemins des scripts MATLAB pour les données d'entraînement et de test
matlab_script_train = 'dynamics_estimator_hysteresis'
matlab_script_test = 'dynamics_estimator_hysteresis_test'

# Exécuter les scripts MATLAB pour extraire les données
eng.eval(matlab_script_train, nargout=0)
force_data_train = eng.workspace['force_data']
gait_vector_train = eng.workspace['gait_vector']
gait_phases_train = eng.workspace['gait_phases']
ankle_angles_filt_train = eng.workspace['ankle_angles_filt']

eng.eval(matlab_script_test, nargout=0)
force_data_test = eng.workspace['force_data']
gait_vector_test = eng.workspace['gait_vector']
gait_phases_test = eng.workspace['gait_phases']
ankle_angles_filt_test = eng.workspace['ankle_angles_filt_test']

eng.quit()  # Arrêter le moteur MATLAB

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

# Préparation des caractéristiques combinées
def prepare_features(X, z, a_100Hz):
    min_length = min(len(X), len(z), len(a_100Hz))
    X, z, a_100Hz = X[:min_length], z[:min_length], a_100Hz[:min_length]

    # Combinaison des caractéristiques
    features = np.hstack((X, z.reshape(-1, 1), a_100Hz.reshape(-1, 1)))

    # Ajout de transformations polynomiales de degré 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    features_poly = poly.fit_transform(features)
    
    return features_poly

X_combined_train = prepare_features(X_train, z_train, a_train_100Hz)
X_combined_test = prepare_features(X_test, z_test, a_test_100Hz)

# Standardisation des caractéristiques (fit sur les données d'entraînement uniquement)
scaler = StandardScaler()
X_combined_scaled_train = scaler.fit_transform(X_combined_train)
X_combined_scaled_test = scaler.transform(X_combined_test)

# Ajuster les longueurs pour l'alignement avant de créer les séquences
min_length_train = min(len(X_combined_scaled_train), len(y_train))
X_combined_scaled_train = X_combined_scaled_train[:min_length_train]
y_train = y_train[:min_length_train]

min_length_test = min(len(X_combined_scaled_test), len(y_test))
X_combined_scaled_test = X_combined_scaled_test[:min_length_test]
y_test = y_test[:min_length_test]

# Création des séquences avec un décalage
def create_sequences(data, labels, seq_length):
    sequences, label_sequences = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
        # Moyenne de plusieurs points de la séquence pour la sortie
        label_sequences.append(np.mean(labels[i:i + seq_length]))  
    return np.array(sequences), np.array(label_sequences)

seq_length = 15
X_seq_train, y_seq_train = create_sequences(X_combined_scaled_train, y_train, seq_length)
X_seq_test, y_seq_test = create_sequences(X_combined_scaled_test, y_test, seq_length)

# Fonction de perte personnalisée
def custom_loss(y_true, y_pred):
    # Pertes standards (MSE)
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Pénaliser les cas où la prédiction dépasse 100
    penalty = tf.where(y_pred >= 100, tf.cast(tf.keras.backend.abs(y_true - 0), tf.float32), 0.0)
    
    # Somme de la perte MSE et de la pénalité
    return mse_loss + penalty

# Modèle CNN avec LSTM, Dropout et TimeDistributed
model = Sequential([
    Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),  # Régularisation avec dropout
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(100, return_sequences=True),
    TimeDistributed(Dense(50, activation='relu')),
    Flatten(),
    Dense(1, activation='linear')
])

# Compiler le modèle avec la perte personnalisée
model.compile(optimizer=Adam(learning_rate=0.00005), loss=custom_loss, metrics=['mae'])

# Callback pour l'early stopping
early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

# Entraînement du modèle
model.fit(
    X_seq_train, y_seq_train, 
    epochs=200, 
    batch_size=64, 
    validation_split=0.2, 
    callbacks=[early_stopping], 
    verbose=1
)

# Prédictions
y_pred = model.predict(X_seq_test).flatten()

# Calcul des erreurs pour le complémentaire
y_pred_complement = 100 - y_pred
mse_complement = mean_squared_error(y_seq_test, y_pred_complement)
mae_complement = mean_absolute_error(y_seq_test, y_pred_complement)
print(f"Mean Squared Error (complement): {mse_complement:.2f}")
print(f"Mean Absolute Error (complement): {mae_complement:.2f}")

# Tracé de la progression réelle vs prédite avec complémentaire
plt.figure()
plt.plot(y_seq_test, label="True gait progress")
plt.plot(y_pred_complement, label="Complement Prediction")
plt.xlabel("Samples")
plt.ylabel("Progression (%)")
plt.title("Comparaison gait progress (Complement)")
plt.legend()
plt.show()

# Calculer le temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")
