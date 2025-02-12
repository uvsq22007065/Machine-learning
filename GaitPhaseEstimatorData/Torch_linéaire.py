import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Démarrer le chronomètre
start_time = time.time()

# Définir les chemins des fichiers CSV
train_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/test_labels.csv"
test_file_path = "C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning/GaitPhaseEstimatorData/validation_labels.csv"

# Charger les données
data_columns = ['Force', 'Force_Derivative', 'Gait_Progress', 'Phase', 'Angle', 'Angle_Derivative']
train_data = pd.read_csv(train_file_path)[data_columns]
test_data = pd.read_csv(test_file_path)[data_columns]

# Créer un mappage pour convertir les phases en chiffres
phase_mapping = {
    'stance_phase': 0,
    'swing_phase': 1,
    'double_stance': 2  # Ajoutez d'autres phases si nécessaire
}

# Appliquer le mappage à la colonne concernée
train_data['Phase'] = train_data['Phase'].map(phase_mapping)
test_data['Phase'] = test_data['Phase'].map(phase_mapping)

# Préparer les caractéristiques et les labels
X_train = train_data.drop(columns=['Gait_Progress']).values
y_train = train_data['Gait_Progress'].values
X_test = test_data.drop(columns=['Gait_Progress']).values
y_test = test_data['Gait_Progress'].values

# Standardiser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer des graphes pour PyTorch Geometric
def create_graph(data):
    edge_index = []
    for i in range(len(data) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])  # Connexions bidirectionnelles
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    features = torch.tensor(data, dtype=torch.float)
    return Data(x=features, edge_index=edge_index)

train_graph = create_graph(X_train_scaled)
test_graph = create_graph(X_test_scaled)

# Définir le modèle NRAX avec PyTorch Geometric
class NRAXModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NRAXModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=1)  # Agrégation globale
        x = self.fc(x.view(1, -1))  # Mise en forme pour correspondre à fc
        return x

# Initialiser le modèle, la perte et l'optimiseur
input_dim = train_graph.x.shape[1]
hidden_dim = 64
output_dim = 1

model = NRAXModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convertir les labels en tenseurs
y_train_tensor = torch.tensor(y_train, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float)

# Entraîner le modèle
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_graph)
    loss = criterion(output, y_train_tensor.mean())  # Ajuster pour résumer les labels
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Évaluer le modèle
model.eval()
with torch.no_grad():
    y_pred = model(test_graph)
    y_pred = y_pred.squeeze()  # Supprime les dimensions de taille 1
    mse = mean_squared_error(y_test_tensor.numpy(), y_pred.numpy())
    mae = mean_absolute_error(y_test_tensor.numpy(), y_pred.numpy())

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Calculer le temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Tracer les résultats
plt.figure()
plt.plot(y_test, label="True gait progress")
plt.plot(y_pred.numpy(), label="Predicted")
plt.xlabel("Samples")
plt.ylabel("Progression (%)")
plt.title("Comparaison gait progress")
plt.legend()
plt.show()
