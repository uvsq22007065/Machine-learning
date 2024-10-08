import matlab.engine

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
import lightgbm as lgb

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

X_combined = np.hstack((X, z.reshape(-1, 1)))

X_derivative = np.diff(X, axis=0)  # Calcul des dérivées de la force

X_combined = X_combined[:-1]
X_combined = np.hstack((X_combined, X_derivative))  # Combiner avec les autres caractéristiques
X_combined = np.hstack((X_combined, a.reshape(-1, 1)))
X_combined = X_combined[:-1]
X_combined = np.hstack((X_combined, a_derivate.reshape(-1, 1)))
y = y[:-2]

print(X_combined.shape)
print(y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Initialize LightGBM model
model = lgb.LGBMClassifier(n_estimators=100000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

import pandas as pd

# Analyze predicted vs actual values
y_pred_values_count = pd.Series(y_pred).value_counts()
y_pred_values_count_percent = pd.Series(y_pred).value_counts(normalize=True)
y_test_values_count = pd.Series(y_test).value_counts()
y_test_values_count_percent = pd.Series(y_test).value_counts(normalize=True)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM Model Accuracy: {accuracy * 100:.2f}%")
print(y_pred_values_count)
print(y_pred_values_count_percent)
print(y_test_values_count)
print(y_test_values_count_percent)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

class_names = ['FF/MST', 'HO', 'HS', 'MSW']

# Visualisation of the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion matrix")
plt.ylabel('Real classes')
plt.xlabel('Predicted classes')
plt.show()
