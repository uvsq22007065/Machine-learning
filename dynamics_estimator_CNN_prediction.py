import matlab.engine
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
import time

# Démarrer le chronomètre
start_time = time.time()

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the MATLAB script paths for training and testing data
matlab_script_train = 'dynamics_estimator_hysteresis'
matlab_script_test = 'dynamics_estimator_hysteresis_test'

# Run the MATLAB script to extract force data and gait vector for training
eng.eval(matlab_script_train, nargout=0)

# Retrieve force data and gait phases from Matlab workspace for training
force_data_train = eng.workspace['force_data']
gait_vector_train = eng.workspace['gait_vector']
gait_phases_train = eng.workspace['gait_phases']
ankle_angles_filt_train = eng.workspace['ankle_angles_filt']

# Run the MATLAB script to extract force data and gait vector for testing
eng.eval(matlab_script_test, nargout=0)

# Retrieve force data and gait vector from Matlab workspace for testing
force_data_test = eng.workspace['force_data']
gait_vector_test = eng.workspace['gait_vector']
gait_phases_test = eng.workspace['gait_phases']
ankle_angles_filt_test = eng.workspace['ankle_angles_filt_test']

# Stop MATLAB engine
eng.quit()

# Assuming 'force_data', 'gait_phases', and 'gait_vector' are arrays from the MATLAB output
X_train = np.array(force_data_train).reshape(-1, 1)  # Force data as feature
y_train = np.array(gait_phases_train)  # Gait vector as labels (gait phases)
z_train = np.array(gait_vector_train)
a_train = np.array(ankle_angles_filt_train)

# Testing data (to predict)
X_test = np.array(force_data_test).reshape(-1, 1)
z_test = np.array(gait_vector_test)
y_test = np.array(gait_phases_test)
a_test = np.array(ankle_angles_filt_test)

# Flatten arrays if needed
y_train = y_train.flatten()
z_train = z_train.flatten()
a_train = a_train.flatten()
z_test = z_test.flatten()
a_test = a_test.flatten()

# Encode gait phases (y) into integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)  # Transform gait phases to integers

# Find the minimum length among X, y, and z
min_length = min(len(X_train), len(y_train), len(z_train), len(a_train))

# Truncate X, y, z, and a to the minimum length
X_train = X_train[:min_length]
y_train = y_train[:min_length]
z_train = z_train[:min_length]
a_train = a_train[:min_length]

a_train = a_train[:-1]
a_train_derivate = np.diff(a_train, axis=0)

X_combined_train = np.hstack((X_train, z_train.reshape(-1, 1)))
X_derivative_train = np.diff(X_train, axis=0)  # Calcul des dérivées de la force

X_combined_train = X_combined_train[:-1]
X_combined_train = np.hstack((X_combined_train, X_derivative_train))  # Combiner avec les autres caractéristiques
X_combined_train = np.hstack((X_combined_train, a_train.reshape(-1, 1)))
X_combined_train = X_combined_train[:-1]
X_combined_train = np.hstack((X_combined_train, a_train_derivate.reshape(-1, 1)))
y_train = y_train[:-2]

# Standardize the features
scaler = StandardScaler()
X_combined_scaled_train = scaler.fit_transform(X_combined_train)

# *** Fenêtrage des données pour Conv1D ***
# Fonction pour créer des fenêtres de séquences temporelles
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Créer des fenêtres de 10 pas de temps
seq_length = 10
X_combined_reshaped_train = create_sequences(X_combined_scaled_train, seq_length)
y_seq_test = y_test[seq_length-1:]  # Ajuster les étiquettes pour correspondre à la longueur des séquences
y_seq_train = y_train[seq_length-1:]

# Vérification des nouvelles formes
print(f"Shape of X_combined_reshaped_train: {X_combined_reshaped_train.shape}")
print(f"Shape of y_seq_test: {y_seq_test.shape}")

# *** Fin du fenêtrage des données pour Conv1D ***
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of z_test: {z_test.shape}")
print(f"Shape of a_test: {a_test.shape}")

# Similarly, process test data
# Truncate X, y, z, and a to the minimum length
min_length_test = min(len(X_test), len(z_test), len(a_test))

X_test = X_test[:min_length_test]
z_test = z_test[:min_length_test]
a_test = a_test[:min_length_test]

print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of z_test: {z_test.shape}")
print(f"Shape of a_test: {a_test.shape}")

a_test = a_test[:-1]
a_test_derivate = np.diff(a_test, axis=0)

X_combined_test = np.hstack((X_test, z_test.reshape(-1, 1)))
X_derivative_test = np.diff(X_test, axis=0)  # Calcul des dérivées de la force
print(f"Shape of X_combined_test: {X_combined_test.shape}")
X_combined_test = X_combined_test[:-1]
X_combined_test = np.hstack((X_combined_test, X_derivative_test))  # Combiner avec les autres caractéristiques
print(f"Shape of X_combined_test: {X_combined_test.shape}")
X_combined_test = np.hstack((X_combined_test, a_test.reshape(-1, 1)))
print(f"Shape of X_combined_test: {X_combined_test.shape}")
X_combined_test = X_combined_test[:-1]
X_combined_test = np.hstack((X_combined_test, a_test_derivate.reshape(-1, 1)))
print(f"Shape of X_combined_test: {X_combined_test.shape}")

# Standardize the features
scaler = StandardScaler()
X_combined_scaled_test = scaler.fit_transform(X_combined_test)


# Créer des fenêtres de 10 pas de temps
seq_length = 10
X_combined_reshaped_test = create_sequences(X_combined_scaled_test, seq_length)

# Build the CNN model
model = Sequential()
model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_combined_reshaped_train.shape[1], X_combined_reshaped_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Assuming 4 phases of gait

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_combined_reshaped_train, y_seq_train, epochs=15, batch_size=32)

# Predict on the test set
y_pred = model.predict(X_combined_reshaped_test)
y_pred = y_pred.argmax(axis=1)

min_length_test2 = min(len(y_test), len(y_pred))
y_pred = y_pred[:min_length_test]
y_test = y_test[:min_length_test]
y_seq_test = y_test[:-1]
y_seq_test = y_test[11:]

y_seq_test = label_encoder.fit_transform(y_seq_test)  # Transform gait phases to integers

# Confusion matrix
cm = confusion_matrix(y_seq_test, y_pred)
class_names = label_encoder.inverse_transform([0, 1, 2, 3])  # Convert encoded labels back to original

# Evaluate the model
accuracy = accuracy_score(y_seq_test, y_pred)
print(f"CNN Model Accuracy: {accuracy * 100:.2f}%")

# Arrêter le chronomètre
end_time = time.time()

# Calculer le temps écoulé
elapsed_time = end_time - start_time
print(f"Temps d'exécution: {elapsed_time} secondes")

# Visualisation de la matrice de confusion
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion matrix")
plt.ylabel('Real classes')
plt.xlabel('Predicted classes')
plt.show()
