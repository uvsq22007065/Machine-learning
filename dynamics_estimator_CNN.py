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

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define the path to the MATLAB file and data files
matlab_script = 'dynamics_estimator_hysteresis'

# Run the MATLAB script to extract force data and gait vector
eng.eval(matlab_script, nargout=0)

# Retrieve force data and gait phases from Matlab workspace
force_data = eng.workspace['force_data']
gait_vector = eng.workspace['gait_vector']
gait_phases = eng.workspace['gait_phases']
ankle_angles_filt = eng.workspace['ankle_angles_filt']

# Stop MATLAB engine
eng.quit()

# Assuming 'force_data', 'gait_phases', and 'gait_vector' are arrays from the MATLAB output
X = np.array(force_data).reshape(-1, 1)  # Force data as feature
y = np.array(gait_phases)  # Gait vector as labels (gait phases)
z = np.array(gait_vector)
a = np.array(ankle_angles_filt)

# Flatten arrays if needed
y = y.flatten()
z = z.flatten()
a = a.flatten()

# Encode gait phases (y) into integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Transform gait phases to integers

# Find the minimum length among X, y, and z
min_length = min(len(X), len(y), len(z), len(a))

# Truncate X, y, z, and a to the minimum length
X = X[:min_length]
y = y[:min_length]
z = z[:min_length]
a = a[:min_length]

a = a[:-1]
a_derivate = np.diff(a, axis=0)

X_combined = np.hstack((X, z.reshape(-1, 1)))
X_derivative = np.diff(X, axis=0)  # Calcul des dérivées de la force

X_combined = X_combined[:-1]
X_combined = np.hstack((X_combined, X_derivative))  # Combiner avec les autres caractéristiques
X_combined = np.hstack((X_combined, a.reshape(-1, 1)))
X_combined = X_combined[:-1]
X_combined = np.hstack((X_combined, a_derivate.reshape(-1, 1)))
y = y[:-2]

# Standardize the features
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# *** Fenêtrage des données pour Conv1D ***
# Fonction pour créer des fenêtres de séquences temporelles
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Créer des fenêtres de 10 pas de temps
seq_length = 10
X_combined_reshaped = create_sequences(X_combined_scaled, seq_length)
y_seq = y[seq_length-1:]  # Ajuster les étiquettes pour correspondre à la longueur des séquences

# Vérification des nouvelles formes
print(f"Shape of X_combined_reshaped: {X_combined_reshaped.shape}")
print(f"Shape of y_seq: {y_seq.shape}")

# *** Fin du fenêtrage des données pour Conv1D ***

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined_reshaped, y_seq, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Assuming 4 phases of gait

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"CNN Model Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = label_encoder.inverse_transform([0, 1, 2, 3])  # Convert encoded labels back to original

# Visualisation de la matrice de confusion
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion matrix")
plt.ylabel('Real classes')
plt.xlabel('Predicted classes')
plt.show()
