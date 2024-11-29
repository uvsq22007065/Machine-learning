#!/usr/bin/env python3

import statistics
import rospy, rospkg, rosbag
from std_msgs.msg import Float64, Int16
import os
import numpy as np
import csv
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import torch
import torch_geometric
from torch_geometric.utils import dense_to_sparse, grid
import torch_geometric.utils as utils


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
        # ROS setup
        rospy.init_node('gait_phase_estimator', anonymous=True)

        # Paths to models and training data
        self.patient = rospy.get_param("patient", "test")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ankle_exoskeleton')

        self.labels_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_labels3.csv")
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_modelR3.pkl")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"2kmph.bag")

        # Variables to track state
        self.modelLoaded = False
        self.model = []
        self.angleUpdated = False
        self.forceUpdated = False

        # Variables for derivatives
        self.last_angle_timestamp = None
        self.last_force_timestamp = None

        # Variable for estimation
        self.samples_size = 2
        self.data_sequence = deque(maxlen=self.samples_size)
        self.ankle_angle = None
        self.ground_force = None
        self.smoothed_estimated_phase = 0

        # ROS Subscribers and Publishers
        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)
        self.gait_ptg_pub = rospy.Publisher('/gait_percentage_GCN3', Int16, queue_size=2)
        self.force_dt_pub = rospy.Publisher('/ground_force_derivative', Float64, queue_size=2)
        self.angle_dt_pub = rospy.Publisher('/angle_force_derivative', Float64, queue_size=2)
        self.phase_pub = rospy.Publisher('/stance_swing_phase_GCN3', Int16, queue_size=2)  # New publisher for stance/swing phase
        self.current_phase = []

    def ankle_angle_callback(self, msg):
        current_angle = msg.data
        if self.ankle_angle is not None:
            self.ankle_angle_derivative = self.calculate_derivative(current_angle, self.ankle_angle)
        else:
            self.ankle_angle_derivative = 0  # Initial derivative or no previous data
        self.ankle_angle = current_angle
        self.angleUpdated = True

    def ground_force_callback(self, msg):
        current_force = msg.data
        if self.ground_force is not None:
            self.ground_force_derivative = self.calculate_derivative(current_force, self.ground_force)
        else:
            self.ground_force_derivative = 0  # Initial derivative or no previous data
        self.ground_force = current_force
        self.forceUpdated = True
    
    def calculate_derivative(self, current_value, previous_value):
        """Calculate the derivative using finite difference method."""
        return (current_value - previous_value)

    def offline_phase_estimator(self, time, interpolated_forces):
        stance_mask = interpolated_forces > 0.1  # True for stance, False for swing
        gait_phases = []
        gait_progress = []
        start_index = 0
        in_stance_phase = stance_mask[0]
        phase_boundaries = []

        for i in range(1, len(stance_mask)):
            if stance_mask[i] != in_stance_phase:
                phase_boundaries.append((start_index, i - 1, in_stance_phase))
                start_index = i
                in_stance_phase = stance_mask[i]

        phase_boundaries.append((start_index, len(stance_mask) - 1, in_stance_phase))
        gait_cycles = []
        i = 0
        while i < len(phase_boundaries) - 1:
            start_stance, end_stance, is_stance = phase_boundaries[i]
            start_swing, end_swing, is_swing = phase_boundaries[i + 1]
            if is_stance and not is_swing:
                gait_cycles.append((start_stance, end_stance, start_swing, end_swing))
                i += 2
            else:
                i += 1

        if len(gait_cycles) > 2:
            gait_cycles = gait_cycles[1:-1]

        start_time = time[gait_cycles[0][0]]
        end_time = time[gait_cycles[-1][3]]

        for start_stance, end_stance, start_swing, end_swing in gait_cycles:
            stance_duration = start_swing - start_stance
            swing_duration = end_swing - end_stance
            stance_progress = np.linspace(0, 60, stance_duration, endpoint=False)
            swing_progress = np.linspace(60, 100, swing_duration, endpoint=False)

            gait_phases.extend(['stance_phase'] * stance_duration)
            gait_phases.extend(['swing_phase'] * swing_duration)
            gait_progress.extend(stance_progress)
            gait_progress.extend(swing_progress)

        gait_phases = np.array(gait_phases)
        gait_progress = np.array(gait_progress)

        return gait_phases, gait_progress, start_time, end_time

        # Construire un graphe pour les données d'entraînement
    def create_graph_data(self, X, y):
        from torch_geometric.data import Data
        
        # Exemple simple de graphe fully-connected
        num_nodes = X.shape[0]
        self.edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
        self.edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)  # Graphe non orienté

        return Data(x=torch.tensor(X, dtype=torch.float), y=torch.tensor(y, dtype=torch.float), edge_index=self.edge_index)

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

        (gait_phases, gait_progress, start_time, end_time) = self.offline_phase_estimator(time, interpolated_forces)

        mask = (time >= start_time) & (time <= end_time)
        adjusted_time = time[mask]
        adjusted_force = interpolated_forces[mask]
        adjusted_angle = interpolated_angles[mask]
        adjusted_force_derivatives = force_derivatives[mask]
        adjusted_angle_derivatives = angle_derivatives[mask]

        rospy.loginfo(f"Saving gait data to {self.labels_path}...")
        with open(self.labels_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Force', 'Force_Derivative', 'Angle', 'Angle_Derivative', 'Gait_Progress', 'Phase'])
            for i in range(len(gait_phases)):
                writer.writerow([
                    adjusted_time[i],
                    adjusted_force[i],
                    adjusted_force_derivatives[i],
                    adjusted_angle[i],
                    adjusted_angle_derivatives[i],
                    gait_progress[i],
                    gait_phases[i]
                ])

        rospy.loginfo(f"Gait data saved successfully.")

        X = np.column_stack((adjusted_force, adjusted_force_derivatives, adjusted_angle, adjusted_angle_derivatives))
        y = np.array(gait_progress)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the model
        input_dim = X_train.shape[1]
        self.model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
    
        X_train_tensor = torch.tensor(X_train, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)

        rospy.loginfo("Training the GCN model...")
        batch_size = 64  # Ajuster selon la mémoire disponible
        num_batches = (X_train_tensor.size(0) // batch_size + 1) - 1

        for epoch in range(10):  # Réduisez le nombre d'epochs pour tester
            for i in range(num_batches - 1):
                optimizer.zero_grad()
                batch_indices = torch.arange(i * batch_size, min((i + 1) * batch_size, X_train_tensor.size(0)))
                batch_set = set(batch_indices.tolist())

                # Création des données de graphe
                graph_data = self.create_graph_data(X_train_tensor.numpy(), y_train_tensor.numpy())

                edge_index = graph_data.edge_index
                mask = [(src in batch_set) and (dst in batch_set) for src, dst in edge_index.t().tolist()]
                filtered_edge_index = edge_index[:, torch.tensor(mask)]

                node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(batch_indices.tolist())}
                mapped_edge_index = torch.tensor(
                    [[node_mapping[src.item()], node_mapping[dst.item()]]
                     for src, dst in filtered_edge_index.t() if src.item() in node_mapping and dst.item() in node_mapping],
                    dtype=torch.long
                ).t()

                predictions = self.model(X_train_tensor[batch_indices], mapped_edge_index)
                loss = criterion(predictions, y_train_tensor[batch_indices])
                loss.backward()
                optimizer.step()

            self.model.train()

        # Test the model
        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)

        with torch.no_grad():
            self.edge_index_test, _ = dense_to_sparse(torch.eye(X_test_tensor.size(0)))
            predictions = self.model(X_test_tensor, self.edge_index_test)
            mse = mean_squared_error(y_test_tensor.numpy(), predictions.numpy())
            rospy.loginfo(f"Test Mean Squared Error: {mse}")

        # Save the trained model
        rospy.loginfo(f"Saving model to {self.model_path}...")
        torch.save(self.model.state_dict(), self.model_path)
        rospy.loginfo("Model saved successfully.")

        # Reload the model to verify
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.modelLoaded = True

    def mean_filter(self, predictions, window_size, anomaly_threshold=50):
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor([predictions], dtype=torch.float)

        predictions = predictions.clone()  # Copie locale pour éviter de modifier directement l'entrée

        if len(predictions) < 2:
            return predictions

        for i in range(1, len(predictions)):
            if predictions[i] < 0:
                predictions[i] = 0
            elif predictions[i] > 100:
                predictions[i] = 100

        # Gérer les anomalies
        for i in range(2, len(predictions)):
            if i >= 5 and torch.abs(predictions[i - 5] - predictions[i]) > anomaly_threshold:
                # Cas d'un saut anormal
                predictions[i] = 0
                predictions = predictions[i:]  # Réinitialiser la séquence pour redémarrer
                break
            elif predictions[i - 1] > predictions[i]:
                # Cas d'une régression (diminution anormale)
                predictions[i] = predictions[i - 1]
                predictions[i] = statistics.mean(predictions)
            elif predictions[i] - predictions[i - 1] > 5 * (predictions[i - 1] - predictions[i - 2]):
                # Cas d'une augmentation trop rapide
                predictions[i] = predictions[i - 1]
                predictions[i] = statistics.mean(predictions)

        # Convolution pour le lissage
        kernel = torch.ones(window_size) / window_size
        smoothed_predictions = torch.nn.functional.conv1d(
            predictions.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=window_size // 2
        ).squeeze()

        return smoothed_predictions

    def estimate_phase(self):
        if self.modelLoaded and self.angleUpdated and self.forceUpdated:
            # Préparer les entrées du graphe
            current_input = np.array([[
                self.ground_force,
                self.ground_force_derivative,
                self.ankle_angle,
                self.ankle_angle_derivative
            ]])
        
            graph_data = self.create_graph_data(current_input, np.zeros(1))  # Labels fictifs
            graph_data.x = torch.tensor(current_input, dtype=torch.float)
        
            # Prédiction
            self.model.eval()
            with torch.no_grad():
                predicted_phase = self.model(graph_data.x, graph_data.self.edge_index).item()
                # Appliquer la fonction mean_filter pour lisser et gérer les anomalies
                self.smoothed_estimated_phase = self.mean_filter(
                    torch.tensor([predicted_phase]), self.samples_size
                )[-1]

            phase_indicator = Int16()
            phase_indicator.data = 100 if self.ground_force == 0 else 0
            self.phase_pub.publish(phase_indicator)

            # Publier les résultats
            self.gait_ptg_pub.publish(int(self.smoothed_estimated_phase))

            self.angleUpdated = False
            self.forceUpdated = False

    def run(self):
        rate = rospy.Rate(200)

        if os.path.exists(self.model_path):
            rospy.loginfo(f"Model found for patient {self.patient}. Loading the model...")
            self.model = GCN(input_dim=4, hidden_dim=64, output_dim=1)  # Replace dimensions if needed
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()  # Set to evaluation mode
            self.modelLoaded = True
        else:
            rospy.logwarn(f"Model not found for patient {self.patient}. Training a new model.")
            res = self.train_model()
            if res == 0:
                rospy.signal_shutdown("Model was not found")

        rospy.loginfo(f"Estimating phase for patient {self.patient} using model {self.model_path}...")
        while not rospy.is_shutdown():
            if self.modelLoaded:
                self.estimate_phase()
            rate.sleep()

if __name__ == '__main__':
    try:
        estimator = GaitPhaseEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
