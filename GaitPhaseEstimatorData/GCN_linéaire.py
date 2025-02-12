import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

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

# Création de la matrice d'adjacence pour les connexions entre capteurs (exemple simplifié)
n_nodes = features_train_scaled.shape[0]
edge_index = torch.tensor([
    [i, i + 1] for i in range(n_nodes - 1)
] + [
    [i + 1, i] for i in range(n_nodes - 1)
], dtype=torch.long).t().contiguous()

# Conversion en tenseurs PyTorch
x_train = torch.tensor(features_train_scaled, dtype=torch.float)
y_train = torch.tensor(labels_train, dtype=torch.float).unsqueeze(1)
x_test = torch.tensor(features_test_scaled, dtype=torch.float)
y_test = torch.tensor(labels_test, dtype=torch.float).unsqueeze(1)

# Créer les graphes d'entraîment et de test
graph_train = Data(x=x_train, edge_index=edge_index, y=y_train)
graph_test = Data(x=x_test, edge_index=edge_index, y=y_test)

# Définir le modèle GCN
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialiser le modèle
input_dim = x_train.size(1)
hidden_dim = 32
output_dim = 1
model = GCNModel(input_dim, hidden_dim, output_dim)

# Définir l'optimiseur et la fonction de perte
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Entraîner le modèle
epochs = 100000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(graph_train.x, graph_train.edge_index)
    loss = criterion(output, graph_train.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Évaluer le modèle
model.eval()
with torch.no_grad():
    predictions = model(graph_test.x, graph_test.edge_index)
    mse = mean_squared_error(graph_test.y.numpy(), predictions.numpy())
    mae = mean_absolute_error(graph_test.y.numpy(), predictions.numpy())
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

# Temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Tracé des résultats
plt.figure()
plt.plot(graph_test.y.numpy(), label="True")
plt.plot(predictions.numpy(), label="Predicted")
plt.xlabel("Samples")
plt.ylabel("Gait Progress")
plt.title("True vs Predicted Gait Progress")
plt.legend()
plt.show()
