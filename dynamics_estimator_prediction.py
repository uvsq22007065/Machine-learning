import matlab.engine

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

# Run the MATLAB script to extract force data and gait vector for testing
eng.eval(matlab_script_test, nargout=0)

# Retrieve force data and gait vector from Matlab workspace for testing
force_data_test = eng.workspace['force_data']
gait_vector_test = eng.workspace['gait_vector']
gait_phases_test = eng.workspace['gait_phases']

# Stop MATLAB engine
eng.quit()

# Continue in Python to work with the data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'force_data_train', 'gait_phases_train', and 'gait_vector_train' are arrays from MATLAB output
# Training data
X_train = np.array(force_data_train).reshape(-1, 1)  # Force data as feature
y_train = np.array(gait_phases_train)  # Gait vector as labels (gait phases)
z_train = np.array(gait_vector_train)

# Testing data (to predict)
X_test = np.array(force_data_test).reshape(-1, 1)
z_test = np.array(gait_vector_test)
y_test = np.array(gait_phases_test)

# Flatten arrays if needed
y_train = y_train.flatten()
z_train = z_train.flatten()
z_test = z_test.flatten()

# Ensure data lengths match for training data
min_length_train = min(len(X_train), len(y_train), len(z_train))
X_train = X_train[:min_length_train]
y_train = y_train[:min_length_train]
z_train = z_train[:min_length_train]

# Combine force data and gait vector for training data
X_combined_train = np.hstack((X_train, z_train.reshape(-1, 1)))

# Compute derivatives for training data
X_derivative_train = np.diff(X_train, axis=0)

# Adjust training data to account for derivative
X_combined_train = X_combined_train[:-1]
X_combined_train = np.hstack((X_combined_train, X_derivative_train))
y_train = y_train[:-1]

# Similarly, process test data
min_length_test = min(len(X_test), len(z_test))
X_test = X_test[:min_length_test]
z_test = z_test[:min_length_test]

# Combine test force data and gait vector
X_combined_test = np.hstack((X_test, z_test.reshape(-1, 1)))

# Compute derivatives for test data
X_derivative_test = np.diff(X_test, axis=0)

# Adjust test data to account for derivative
X_combined_test = X_combined_test[:-1]
X_combined_test = np.hstack((X_combined_test, X_derivative_test))

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100)

# Train the model on the training set
model.fit(X_combined_train, y_train)

# Predict on the test set
y_pred = model.predict(X_combined_test)

# Evaluate the model
# Since we do not have true labels for the test data, we'll just output the predictions
print("Predictions for test set:")
print(y_pred)

# Output the counts and proportions of the predicted phases
y_pred_values_count = pd.Series(y_pred).value_counts()
y_pred_values_count_percent = pd.Series(y_pred).value_counts(normalize=True)

print(y_pred_values_count)
print(y_pred_values_count_percent)

min_length_test2 = min(len(y_test), len(y_pred))
y_pred = y_pred[:min_length_test]
y_test = y_test[:min_length_test]
y_test = y_test[:-1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Supposons que y_test sont les vraies classes et y_pred sont les classes prédites
cm = confusion_matrix(y_test, y_pred)

class_names = ['FF/MST', 'HO', 'HS', 'MSW']

# Visualisation de la matrice de confusion
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion matrix")
plt.ylabel('Real classes')
plt.xlabel('Predicted classes')
plt.show()
