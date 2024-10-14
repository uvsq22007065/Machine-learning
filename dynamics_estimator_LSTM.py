import matlab.engine
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Démarrer le chronomètre
start_time = time.time()

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

# Find the minimum length among X, y, and z
min_length = min(len(X), len(y), len(z), len(a))

# Truncate X, y, and z to the minimum length
X = X[:min_length]
y = y[:min_length]
z = z[:min_length]
a = a[:min_length]

a = a[:-1]
a_derivate = np.diff(a, axis=0)

# Combine the features
X_combined = np.hstack((X, z.reshape(-1, 1)))

# Calculate the derivative of force data
X_derivative = np.diff(X, axis=0)

# Prepare the combined feature set
X_combined = X_combined[:-1]
X_combined = np.hstack((X_combined, X_derivative))
X_combined = np.hstack((X_combined, a.reshape(-1, 1)))
X_combined = X_combined[:-1]
X_combined = np.hstack((X_combined, a_derivate.reshape(-1, 1)))
y = y[:-2]

# Convert data to correct types if necessary
X_combined = X_combined.astype(np.float32)

# Encoder les étiquettes avec LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined).astype(np.float32)

# Reshape data for LSTM [samples, timesteps, features]
X_combined_reshaped = X_combined_scaled.reshape((X_combined_scaled.shape[0], 1, X_combined_scaled.shape[1]))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined_reshaped, y_encoded, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Assuming 4 phases of gait

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)

# Inverser l'encodage pour obtenir les phases sous forme de chaînes de caractères
y_pred_phases = label_encoder.inverse_transform(y_pred)
y_test_phases = label_encoder.inverse_transform(y_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"LSTM Model Accuracy: {accuracy * 100:.2f}%")

# Count the predicted and true class values
y_pred_values_count = pd.Series(y_pred_phases).value_counts()
y_pred_values_count_percent = pd.Series(y_pred_phases).value_counts(normalize=True)
y_test_values_count = pd.Series(y_test_phases).value_counts()
y_test_values_count_percent = pd.Series(y_test_phases).value_counts(normalize=True)

print("Predicted values count:")
print(y_pred_values_count)
print(y_pred_values_count_percent)
print("True values count:")
print(y_test_values_count)
print(y_test_values_count_percent)

# Calcul du temps écoulé
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Temps d'exécution: {elapsed_time} secondes")

# Confusion Matrix
cm = confusion_matrix(y_test_phases, y_pred_phases)
class_names = label_encoder.classes_

# Visualize confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
