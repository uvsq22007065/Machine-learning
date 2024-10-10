import matlab.engine
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

# Continue in Python to work with the data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.interpolate import interp1d

# Assuming 'force_data', 'gait_phases', and 'gait_vector' are arrays from the MATLAB output
X = np.array(force_data).reshape(-1, 1)  # Force data as feature
y = np.array(gait_phases)  # Gait vector as labels (gait phases)
z = np.array(gait_vector)
a = np.array(ankle_angles_filt)

# Flatten arrays if needed
y = y.flatten()
z = z.flatten()
a = a.flatten()

# Interpolation de la variable a_train (à 60 Hz) vers 100 Hz
t_60Hz = np.linspace(0, len(a) / 60, len(a))
t_100Hz = np.linspace(0, len(a) / 60, int(len(a) * 100 / 60))
interp_a = interp1d(t_60Hz, a, kind='linear')
a_100Hz = interp_a(t_100Hz)

# Find the minimum length among X, y, and z
min_length = min(len(X), len(y), len(z), len(a_100Hz))

# Truncate X, y, and z to the minimum length
X = X[:min_length]
y = y[:min_length]
z = z[:min_length]
a_100Hz = a_100Hz[:min_length]

a_100Hz = a_100Hz[:-1]
a_100Hz_derivate = np.diff(a_100Hz, axis=0)

# Combine features (force data, gait vector, ankle angles, and their derivatives)
X_combined = np.hstack((X, z.reshape(-1, 1)))
X_derivative = np.diff(X, axis=0)  # Calculate the derivatives of the force
X_combined = X_combined[:-1]
X_combined = np.hstack((X_combined, X_derivative))  # Combine with other features
X_combined = np.hstack((X_combined, a_100Hz.reshape(-1, 1)))
X_combined = X_combined[:-1]
X_combined = np.hstack((X_combined, a_100Hz_derivate.reshape(-1, 1)))
y = y[:-2]

# Encode the labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize the data
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)

# Initialize XGBoost model with basic parameters
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Decode the predicted labels back to their original form
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# Supposons que "y" soit la variable qui contient les étiquettes (les classes)
y_pred_values_count = pd.Series(y_pred_labels).value_counts()
y_pred_values_count_percent = pd.Series(y_pred_labels).value_counts(normalize=True)
y_test_values_count = pd.Series(y_test_labels).value_counts()
y_test_values_count_percent = pd.Series(y_test_labels).value_counts(normalize=True)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(y_pred_values_count)
print(y_pred_values_count_percent)
print(y_test_values_count)
print(y_test_values_count_percent)

# Confusion matrix and visualization
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)

class_names = ['FF/MST', 'HO', 'HS', 'MSW']

# Arrêter le chronomètre
end_time = time.time()

# Calculer le temps écoulé
elapsed_time = end_time - start_time
print(f"Temps d'exécution: {elapsed_time} secondes")

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion matrix")
plt.ylabel('Real classes')
plt.xlabel('Predicted classes')
plt.show()
