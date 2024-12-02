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

        self.labels_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_labelsGCN.csv")
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_modelGCN.pkl")
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
        self.samples_size = 5
        self.data_sequence = deque(maxlen=self.samples_size)
        self.ankle_angle = None
        self.ground_force = None
        self.smoothed_estimated_phase = 0

        # ROS Subscribers and Publishers
        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)
        self.gait_ptg_pub = rospy.Publisher('/gait_percentage_GCN', Int16, queue_size=2)
        self.force_dt_pub = rospy.Publisher('/ground_force_derivative', Float64, queue_size=2)
        self.angle_dt_pub = rospy.Publisher('/angle_force_derivative', Float64, queue_size=2)
        self.phase_pub = rospy.Publisher('/stance_swing_phase_GCN', Int16, queue_size=2)  # New publisher for stance/swing phase
        self.current_phase = []
        self.prediction_buffer = []

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
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Graphe non orienté

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
        for epoch in range(20000):
            self.model.train()
            optimizer.zero_grad()

            # Convert dense adjacency matrix to edge_index
            A = torch.eye(X_train_tensor.size(0))  # Simulated adjacency matrix
            edge_index, _ = dense_to_sparse(A)  # Convert to sparse format

            predictions = self.model(X_train_tensor, edge_index)
            loss = criterion(predictions, y_train_tensor)

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                rospy.loginfo(f"Epoch {epoch}, Loss: {loss.item()}")

            # Early stopping
            if loss.item() < 15:
                break

        # Test the model
        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)

        with torch.no_grad():
            edge_index_test, _ = dense_to_sparse(torch.eye(X_test_tensor.size(0)))
            predictions = self.model(X_test_tensor, edge_index_test)
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
        import torch.nn.functional as F  # Import local pour F.conv1d

        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, dtype=torch.float)

        predictions = predictions.clone()  # Évite la modification directe
        predictions = torch.clamp(predictions, 0, 100)  # Limite entre 0 et 100

        # Détection des anomalies et régressions AVANT lissage
        for i in range(len(predictions)):
            delta_prev = torch.abs(predictions[i] - predictions[i - 1])

            # Détection des anomalies
            if delta_prev > anomaly_threshold:
                rospy.logwarn(f"Anomalie détectée à l'indice {i}: saut anormal.")
                predictions[i:] = 0
                break

            # Détection des régressions
            elif predictions[i] < predictions[i - 1] or predictions[i] - predictions[i - 1] > 2 * predictions[i - 1] - predictions[i - 2]:
                rospy.loginfo(f"Correction à l'indice {i}: régression détectée ({predictions[i]} < {predictions[i - 1]}).")
                predictions[i] = predictions[i - 1]  # Forcer la continuité
                predictions[i] = torch.mean(predictions).item()

        # Lissage par convolution
        kernel = torch.ones(window_size) / window_size
        predictions_padded = F.pad(
            predictions.unsqueeze(0).unsqueeze(0),
            (window_size // 2, window_size // 2),
            mode='constant',
            value=0
        )
        smoothed_predictions = F.conv1d(
            predictions_padded,
            kernel.unsqueeze(0).unsqueeze(0)
        ).squeeze()

        # Correction pour atténuation
        correction_factor = window_size / (window_size - (window_size // 2))
        smoothed_predictions *= correction_factor

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
                predicted_phase = self.model(graph_data.x, graph_data.edge_index).item()
                # Ajouter la prédiction au buffer
                self.prediction_buffer.append(predicted_phase)
                if len(self.prediction_buffer) > self.samples_size:
                    self.prediction_buffer.pop(0)

                if len(self.prediction_buffer) == self.samples_size:
                    smoothed_predictions = self.mean_filter(
                    torch.tensor(self.prediction_buffer), self.samples_size
                    )
                    self.smoothed_estimated_phase = smoothed_predictions[-1]

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
