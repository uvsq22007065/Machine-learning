import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Démarrer le chronomètre
start_time = time.time()

# Chemins des fichiers CSV
train_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv"
test_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv"

# Lire les fichiers CSV
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Extraire les données
features_train = train_data[['Force', 'Force_Derivative', 'Angle', 'Angle_Derivative']].values
labels_train = train_data['Gait_Progress'].values
features_test = test_data[['Force', 'Force_Derivative', 'Angle', 'Angle_Derivative']].values
labels_test = test_data['Gait_Progress'].values

# Standardisation des caractéristiques
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Création de séquences temporelles pour LSTM
def create_sequences(data, labels, seq_length):
    sequences, label_sequences = [], []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
        label_sequences.append(labels[i + seq_length - 1])
    return np.array(sequences), np.array(label_sequences)

seq_length = 130
X_seq_train, y_seq_train = create_sequences(features_train_scaled, labels_train, seq_length)
X_seq_test, y_seq_test = create_sequences(features_test_scaled, labels_test, seq_length)

# Conversion en tenseurs PyTorch
X_seq_train = torch.tensor(X_seq_train, dtype=torch.float)
y_seq_train = torch.tensor(y_seq_train, dtype=torch.float)
X_seq_test = torch.tensor(X_seq_test, dtype=torch.float)
y_seq_test = torch.tensor(y_seq_test, dtype=torch.float)

# Créer des DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_seq_train, y_seq_train)
test_dataset = TensorDataset(X_seq_test, y_seq_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Définir le modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn contient les activations finales des couches cachées
        hn = hn[-1]  # Dernière couche cachée
        out = self.fc(hn)
        return out

# Initialiser le modèle
input_dim = X_seq_train.size(2)
hidden_dim = 64
output_dim = 1
num_layers = 2
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

# Définir l'optimiseur et la fonction de perte
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Entraîner le modèle
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Évaluer le modèle
model.eval()
predictions = []
true_values = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch).squeeze()
        predictions.append(preds)
        true_values.append(y_batch)

predictions = torch.cat(predictions).numpy()
true_values = torch.cat(true_values).numpy()

mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Tracé des résultats
plt.figure()
plt.plot(true_values, label="True")
plt.plot(predictions, label="Predicted")
plt.xlabel("Samples")
plt.ylabel("Gait Progress")
plt.title("True vs Predicted Gait Progress")
plt.legend()
plt.show()
