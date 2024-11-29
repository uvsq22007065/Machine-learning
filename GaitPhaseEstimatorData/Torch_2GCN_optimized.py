#!/usr/bin/env python3

import statistics
import rospy, rospkg, rosbag
from std_msgs.msg import Float64, Int16
import os
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

# Configuration pour GPU si disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)  # Activation
        x = self.conv2(x, edge_index)
        return x

class GaitPhaseEstimator:
    def __init__(self):
        rospy.init_node('gait_phase_estimator', anonymous=True)

        # Paths to models and data
        self.patient = rospy.get_param("patient", "test")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ankle_exoskeleton')

        self.labels_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_labels3.csv")
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_modelR3.pkl")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"2kmph.bag")

        self.modelLoaded = False
        self.model = []
        self.angleUpdated = False
        self.forceUpdated = False

        self.last_angle_timestamp = None
        self.last_force_timestamp = None

        self.samples_size = 2
        self.data_sequence = []
        self.ankle_angle = None
        self.ground_force = None
        self.smoothed_estimated_phase = 0
        self.hidden_dim = rospy.get_param("hidden_dim", 64)
        self.batch_size = rospy.get_param("batch_size", 64)
        self.epochs = rospy.get_param("epochs", 10)

        # ROS Subscribers and Publishers
        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)
        self.gait_ptg_pub = rospy.Publisher('/gait_percentage_GCN3', Int16, queue_size=2)
        self.phase_pub = rospy.Publisher('/stance_swing_phase_GCN3', Int16, queue_size=2)
    
    def ankle_angle_callback(self, msg):
        current_angle = msg.data
        self.ankle_angle_derivative = self.calculate_derivative(current_angle, self.ankle_angle or current_angle)
        self.ankle_angle = current_angle
        self.angleUpdated = True

    def ground_force_callback(self, msg):
        current_force = msg.data
        self.ground_force_derivative = self.calculate_derivative(current_force, self.ground_force or current_force)
        self.ground_force = current_force
        self.forceUpdated = True

    def calculate_derivative(self, current_value, previous_value):
        """Calculate the derivative using finite difference."""
        return current_value - previous_value


    def create_graph_data(self, X, y, k=3):
        # Utilisation de KNN pour connecter chaque nœud à ses k plus proches voisins
        adjacency_matrix = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
        return Data(x=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), edge_index=edge_index)

    def train_model(self):
        rospy.loginfo(f"Training model for patient {self.patient}...")
    
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"No .bag file found for patient {self.patient} in {self.bag_path}. Training cannot proceed.")
            rospy.signal_shutdown("Missing .bag file for training.")
            return

        # Load and preprocess data
        bag = rosbag.Bag(self.bag_path)
        angle_data = []
        vgrf_data = []

        for topic, msg, t in bag.read_messages(topics=['/ankle_joint/angle', '/vGRF']):
            if topic == '/ankle_joint/angle':
                angle_data.append((t.to_sec(), msg.data))
            elif topic == '/vGRF':
                vgrf_data.append((t.to_sec(), msg.data))

        bag.close()
        angle_data = np.array(angle_data)
        vgrf_data = np.array(vgrf_data)

        interpolated_angles = np.interp(vgrf_data[:, 0], angle_data[:, 0], angle_data[:, 1])
        interpolated_forces = vgrf_data[:, 1]
        time = vgrf_data[:, 0] - vgrf_data[0, 0]

        force_derivatives = np.gradient(interpolated_forces)
        angle_derivatives = np.gradient(interpolated_angles)

        # Enregistrement des données
        rospy.loginfo(f"Saving gait data to {self.labels_path}...")
        with open(self.labels_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Force', 'Force_Derivative', 'Angle', 'Angle_Derivative'])
            for i in range(len(time)):
                writer.writerow([
                    time[i], interpolated_forces[i], force_derivatives[i], interpolated_angles[i], angle_derivatives[i]
                ])

        # Préparation des données pour l'entraînement
        X = np.column_stack((interpolated_forces, force_derivatives, interpolated_angles, angle_derivatives))
        y = time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialisation du modèle
        input_dim = X_train.shape[1]
        self.model = GCN(input_dim=input_dim, hidden_dim=self.hidden_dim, output_dim=1).to(device)

        batch_size = self.batch_size
        num_batches = (len(X_train_tensor) + batch_size - 1) // batch_size

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
        
        for epoch in range(self.epochs):  # Réduire ou augmenter selon les ressources
            for i in range(num_batches):
                optimizer.zero_grad()
                start = i * batch_size
                end = min(start + batch_size, X_train_tensor.size(0))
                batch_X = X_train_tensor[start:end]
                batch_y = y_train_tensor[start:end]

                predictions = self.model(batch_X, self.create_edge_index(batch_X.size(0)))
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

            # Validation après chaque epoch
            with torch.no_grad():
                self.model.eval()
                val_predictions = self.model(X_train_tensor, self.create_edge_index(X_train_tensor.size(0)))
                val_loss = criterion(val_predictions, y_train_tensor)
                rospy.loginfo(f"Epoch {epoch+1}: Validation Loss = {val_loss.item()}")
                self.model.train()

        rospy.loginfo("Training completed.")

    def create_edge_index(self, num_nodes):
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
        return torch.cat([edge_index, edge_index.flip(0)], dim=1)

    def mean_filter(self, predictions, window_size, anomaly_threshold=50):
        predictions = torch.tensor(predictions, dtype=torch.float) if not isinstance(predictions, torch.Tensor) else predictions.clone()
        if len(predictions) < 2:
            return predictions

        for i in range(1, len(predictions)):
            if predictions[i] < 0:
                predictions[i] = 0
            elif predictions[i] > 100:
                predictions[i] = 100

            if i >= 5 and torch.abs(predictions[i - 5] - predictions[i]) > anomaly_threshold:
                predictions[i] = 0
                return predictions[i:]  
            elif predictions[i - 1] > predictions[i] or predictions[i] - predictions[i - 1] > 5 * (predictions[i - 1] - predictions[i - 2]):
                predictions[i] = predictions[i - 1]

        kernel = torch.ones(window_size) / window_size
        smoothed_predictions = torch.nn.functional.conv1d(predictions.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=window_size // 2).squeeze()
        return smoothed_predictions

    def estimate_phase(self):
        if self.modelLoaded and self.angleUpdated and self.forceUpdated:
            current_input = np.array([[self.ground_force, self.ground_force_derivative, self.ankle_angle, self.ankle_angle_derivative]])
            graph_data = self.create_graph_data(current_input, np.zeros(1))
            graph_data.x = torch.tensor(current_input, dtype=torch.float)

            self.model.eval()
            with torch.no_grad():
                predicted_phase = self.model(graph_data.x, graph_data.edge_index).item()
                self.smoothed_estimated_phase = self.mean_filter([predicted_phase], self.samples_size)[-1]
            self.gait_ptg_pub.publish(int(self.smoothed_estimated_phase))
            self.angleUpdated = False
            self.forceUpdated = False

    def run(self):
        rate = rospy.Rate(200)
        if os.path.exists(self.model_path):
            self.model = GCN(input_dim=4, hidden_dim=64, output_dim=1).to(device)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            self.modelLoaded = True
        else:
            rospy.signal_shutdown("Model not found.")
        while not rospy.is_shutdown():
            self.estimate_phase()
            rate.sleep()

if __name__ == '__main__':
    try:
        estimator = GaitPhaseEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
