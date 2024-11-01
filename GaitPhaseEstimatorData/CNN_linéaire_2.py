import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Chemins vers les fichiers
train_file_path = 'C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv'
test_file_path = 'C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv'

# Charger les données d'entraînement
force_data_train = pd.read_csv(train_file_path)['Force'].values
gait_vector_train = pd.read_csv(train_file_path)['Gait_Progress'].values
gait_phases_train = pd.read_csv(train_file_path)['Phase'].values
ankle_angles_filt_train = pd.read_csv(train_file_path)['Angle'].values

# Charger les données de test
force_data_test = pd.read_csv(test_file_path)['Force'].values
gait_vector_test = pd.read_csv(test_file_path)['Gait_Progress'].values
gait_phases_test = pd.read_csv(test_file_path)['Phase'].values
ankle_angles_filt_test = pd.read_csv(test_file_path)['Angle'].values

# Fonction pour calculer les dérivées
def calculate_derivatives(data):
    return np.diff(data, prepend=data[0])

# Calcul des dérivées pour les données d'entraînement
gait_progress_derivative_train = calculate_derivatives(gait_vector_train)
ankle_angle_derivative_train = calculate_derivatives(ankle_angles_filt_train)
vGRF_derivative_train = calculate_derivatives(force_data_train)

# Calcul des dérivées pour les données de test
gait_progress_derivative_test = calculate_derivatives(gait_vector_test)
ankle_angle_derivative_test = calculate_derivatives(ankle_angles_filt_test)
vGRF_derivative_test = calculate_derivatives(force_data_test)

# Préparation des caractéristiques d'entrée et des cibles pour l'entraînement
X_train = np.column_stack([
    ankle_angle_derivative_train,
    vGRF_derivative_train,
    ankle_angles_filt_train,
    force_data_train
])
y_train = gait_vector_train

# Préparation des caractéristiques d'entrée et des cibles pour le test
X_test = np.column_stack([
    ankle_angle_derivative_test,
    vGRF_derivative_test,
    ankle_angles_filt_test,
    force_data_test
])
y_test = gait_vector_test

# Normalisation des données d'entraînement
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# Normalisation des données de test avec les mêmes scalers d'entraînement
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Reshaper les données pour le modèle LSTM
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Fonction de coût pondérée pour accentuer les erreurs dans les transitions
def weighted_mse(y_true, y_pred):
    weights = tf.where((y_true > 0.6) & (y_true < 1.0), 2.0, 1.0)  # Poids doublé pour les transitions
    return MeanSquaredError()(y_true, y_pred) * weights

# Création du modèle LSTM
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=weighted_mse, metrics=['mse'])
    return model

# Validation croisée sur les données d'entraînement
kf = KFold(n_splits=5)
fold = 1
val_losses = []

for train_index, val_index in kf.split(X_train_lstm):
    print(f"Training fold {fold}...")
    X_tr, X_val = X_train_lstm[train_index], X_train_lstm[val_index]
    y_tr, y_val = y_train_scaled[train_index], y_train_scaled[val_index]
    
    model = create_lstm_model((X_tr.shape[1], X_tr.shape[2]))
    
    # Early stopping pour éviter le surajustement
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Entraînement du modèle
    history = model.fit(X_tr, y_tr, 
                        epochs=15, 
                        batch_size=32, 
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping], 
                        verbose=1)
    
    val_loss = min(history.history['val_loss'])
    val_losses.append(val_loss)
    print(f"Fold {fold} - Validation Loss: {val_loss}")
    fold += 1

# Evaluation finale
print("Validation losses for each fold:", val_losses)
print("Average validation loss:", np.mean(val_losses))

# Prédictions sur l'ensemble de test et dénormalisation pour l'affichage
y_pred_scaled = model.predict(X_test_lstm)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# Affichage des résultats
plt.plot(y_true, label="True gait progress")
plt.plot(y_pred, label="Prediction", alpha=0.7)
plt.xlabel("Samples")
plt.ylabel("Progression (%)")
plt.title("Comparaison gait progress (Test set)")
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calcul des erreurs
mse = mean_squared_error(y_seq_test, y_pred)
mae = mean_absolute_error(y_seq_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")