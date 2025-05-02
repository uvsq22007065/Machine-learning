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
# Chemins des fichiers
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Fonction pour charger et préparer les données d'un dossier
def load_and_prepare_data(folder_path):
    # Chemins vers les fichiers nécessaires
    ankle_file = os.path.join(folder_path, "ankle_joint-angle.csv")
    vgrf_file = os.path.join(folder_path, "vGRF.csv")
    gait_progress_file = os.path.join(folder_path, "gait_progress.csv")
    
    # Vérifier que tous les fichiers existent
    if not all(os.path.exists(f) for f in [ankle_file, vgrf_file, gait_progress_file]):
        print(f"Certains fichiers sont manquants dans le dossier {folder_path}")
        return None
    
    # Lire les fichiers CSV
    ankle_data = pd.read_csv(ankle_file)
    vgrf_data = pd.read_csv(vgrf_file)
    gait_progress_data = pd.read_csv(gait_progress_file)
    
    # Renommer les colonnes pour clarté
    ankle_data = ankle_data.rename(columns={"data": "Angle"})
    vgrf_data = vgrf_data.rename(columns={"data": "Force"})
    
    # Vérifier que ankle_data et gait_progress_data ont le même nombre de lignes
    if len(ankle_data) != len(gait_progress_data):
        print(f"Les fichiers ankle_data et gait_progress_data n'ont pas le même nombre de lignes dans {folder_path}")
        return None
    
    # Créer un nouveau DataFrame avec toutes les données nécessaires
    merged_data = pd.DataFrame()
    merged_data['Time'] = ankle_data['Time']
    merged_data['Angle'] = ankle_data['Angle']
    merged_data['Gait_Progress'] = gait_progress_data['Gait_Progress']
    
    # Interpoler les données de force sur les points de temps de ankle_data
    from scipy.interpolate import interp1d
    
    # Créer une fonction d'interpolation pour la force
    force_interp = interp1d(vgrf_data['Time'], vgrf_data['Force'], 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Appliquer l'interpolation pour obtenir les valeurs de force aux mêmes temps que ankle_data
    merged_data['Force'] = force_interp(merged_data['Time'])
    
    # Calculer les dérivées
    merged_data['Angle_Derivative'] = np.gradient(merged_data['Angle'])
    merged_data['Force_Derivative'] = np.gradient(merged_data['Force'])
    
    # Supprimer les lignes avec des valeurs NaN
    merged_data = merged_data.dropna()
    
    print(f"Données préparées pour {folder_path}: {len(merged_data)} lignes")
    return merged_data

# Demander les chemins des dossiers d'entraînement et de test
train_folder = input("Entrez le chemin du dossier contenant les données d'entraînement : ")
test_folder = input("Entrez le chemin du dossier contenant les données de test : ")

# Charger et préparer les données d'entraînement
train_data = load_and_prepare_data(train_folder)

# Charger et préparer les données de test
test_data = load_and_prepare_data(test_folder)

# Vérifier que les données ont été chargées correctement
if train_data is None or test_data is None:
    print("Erreur lors du chargement des données. Veuillez vérifier les dossiers et fichiers.")
    # Définir des variables vides pour éviter les erreurs
    features_train = np.array([[0, 0, 0, 0]])
    labels_train = np.array([0])
    features_test = np.array([[0, 0, 0, 0]])
    labels_test = np.array([0])
else:
    # Extraire features et labels
    features_train = train_data[['Force', 'Force_Derivative', 'Angle', 'Angle_Derivative']].values
    labels_train = train_data['Gait_Progress'].values
    features_test = test_data[['Force', 'Force_Derivative', 'Angle', 'Angle_Derivative']].values
    labels_test = test_data['Gait_Progress'].values
    
    print(f"Données d'entraînement: {len(features_train)} échantillons")
    print(f"Données de test: {len(features_test)} échantillons")

# Standardisation des caractéristiques
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

print("Préparation des données terminée avec succès.")

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
