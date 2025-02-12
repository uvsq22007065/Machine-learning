import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.models import Model
import networkx as nx
import pandas as pd
import time

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

# Créer des graphes pour NRAX
def create_graph(data):
    G = nx.Graph()
    for i, row in enumerate(data):
        G.add_node(i, features=row)
    for i in range(len(data) - 1):
        G.add_edge(i, i + 1)  # Connexions séquentielles simples
    return G

train_graph = create_graph(X_train_scaled)
test_graph = create_graph(X_test_scaled)

# Convertir le graphe en matrice d'adjacence et caractéristiques
train_adj = nx.adjacency_matrix(train_graph).todense()
train_features = np.array([train_graph.nodes[i]['features'] for i in range(len(train_graph))])

test_adj = nx.adjacency_matrix(test_graph).todense()
test_features = np.array([test_graph.nodes[i]['features'] for i in range(len(test_graph))])

# Définir le modèle NRAX
class NRAXLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NRAXLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(NRAXLayer, self).build(input_shape)

    def call(self, inputs):
        adjacency, features = inputs
        aggregated = tf.matmul(adjacency, features)
        transformed = tf.matmul(aggregated, self.kernel)
        return tf.nn.relu(transformed)

# Construire le modèle NRAX
input_adj = Input(shape=(train_adj.shape[0],), name='adjacency_matrix')
input_features = Input(shape=(train_features.shape[1],), name='node_features')

hidden = NRAXLayer(64)([input_adj, input_features])
hidden = Dropout(0.3)(hidden)
output = Dense(1, activation='linear')(hidden)

nrax_model = Model(inputs=[input_adj, input_features], outputs=output)
nrax_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Préparer les données pour l'entraînement
train_adj_expanded = np.expand_dims(train_adj, axis=0)
test_adj_expanded = np.expand_dims(test_adj, axis=0)

# Entraîner le modèle
nrax_model.fit(
    [train_adj_expanded, train_features], y_train,
    epochs=50,
    batch_size=32,
    validation_data=([test_adj_expanded, test_features], y_test),
    verbose=1
)

# Prédictions
y_pred = nrax_model.predict([test_adj_expanded, test_features])

# Calcul des erreurs
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Calculer le temps d'exécution
elapsed_time = time.time() - start_time
print(f"Temps d'exécution: {elapsed_time:.2f} secondes")

# Tracer les résultats
plt.figure()
plt.plot(y_test, label="True gait progress")
plt.plot(y_pred.flatten(), label="Predicted")
plt.xlabel("Samples")
plt.ylabel("Progression (%)")
plt.title("Comparaison gait progress")
plt.legend()
plt.show()
