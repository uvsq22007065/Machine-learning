#!/usr/bin/env python3

import rospy, rospkg, rosbag
from std_msgs.msg import Float64
import os
import numpy as np
import csv
from time import sleep
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten

class GaitPhaseEstimator:
    def __init__(self):
        # ROS setup
        rospy.init_node('gait_phase_estimator', anonymous=True)

        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)

        # Paths to models and training data
        self.patient = rospy.get_param("patient", "test")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ankle_exoskeleton')

        self.labels_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_labels.csv")
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_model.csv")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"{self.patient}.bag")

        # Variables to track state
        self.modelLoaded = False
        self.ankle_angle = None
        self.ground_force = None
        self.learning_model = []

    def ankle_angle_callback(self, msg):
        self.ankle_angle = msg.data

    def ground_force_callback(self, msg):
        self.ground_force = msg.data

    def offline_phase_estimator(self, time, interpolated_forces):
        # Detect stance and swing phases based on force data
        stance_mask = interpolated_forces > 0.1  # True for stance, False for swing

        # Initialize phase and progress arrays
        gait_phases = []
        gait_progress = []

        # Track indices where each stance and swing phase begins and ends
        start_index = 0
        in_stance_phase = stance_mask[0]
        phase_boundaries = []

        # Identify boundaries of each stance and swing phase
        for i in range(1, len(stance_mask)):
            if stance_mask[i] != in_stance_phase:
                phase_boundaries.append((start_index, i - 1, in_stance_phase))
                start_index = i
                in_stance_phase = stance_mask[i]

        # Add the last boundary
        phase_boundaries.append((start_index, len(stance_mask) - 1, in_stance_phase))

        # Find complete gait cycles (stance + swing phase pairs)
        gait_cycles = []
        i = 0
        while i < len(phase_boundaries) - 1:
            # Ensure a stance phase is followed by a swing phase for a complete gait cycle
            start_stance, end_stance, is_stance = phase_boundaries[i]
            start_swing, end_swing, is_swing = phase_boundaries[i + 1]
            if is_stance and not is_swing:
                gait_cycles.append((start_stance, end_stance, start_swing, end_swing))  # Start of stance to end of swing
                i += 2  # Skip to the next possible cycle
            else:
                i += 1  # Continue looking for a complete cycle

        # Discard the first and last complete gait cycles
        if len(gait_cycles) > 2:
            gait_cycles = gait_cycles[1:-1]  # Remove first and last cycles

        start_time = time[gait_cycles[0][0]]
        end_time = time[gait_cycles[-1][3]]
        # Assign gait phases and progress for the remaining cycles
        for start_stance, end_stance, start_swing, end_swing in gait_cycles:
            # Identify if this is a stance or swing phase
            stance_duration = start_swing - start_stance
            swing_duration = end_swing - end_stance

            # 60% for stance phase
            stance_progress = np.linspace(0, 60, stance_duration, endpoint=False)
            swing_progress = np.linspace(60, 100, swing_duration, endpoint=False)

            gait_phases.extend(['stance_phase'] * stance_duration)
            gait_phases.extend(['swing_phase'] * swing_duration)
            gait_progress.extend(stance_progress)
            gait_progress.extend(swing_progress)

        # Convert lists to numpy arrays for easier processing
        gait_phases = np.array(gait_phases)
        gait_progress = np.array(gait_progress)

        return gait_phases, gait_progress, start_time, end_time
        
    # Fenêtrage des données pour Conv1D
    def create_sequences(self, data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)

    def train_model(self):
        rospy.loginfo(f"Training model for patient {self.patient}...")

        # Check if the bag file exists
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"No .bag file found for patient {self.patient} in {self.bag_path}. Training cannot proceed.")
            rospy.signal_shutdown("Missing .bag file for training.")
            return

        # Read bag file and extract data
        bag = rosbag.Bag(self.bag_path)
        angle_data = []
        vgrf_data = []

        for topic, msg, t in bag.read_messages(topics=['/ankle_joint/angle', '/vGRF']):
            if topic == '/ankle_joint/angle':
                angle_data.append((t.to_sec(), msg.data))
            elif topic == '/vGRF':
                vgrf_data.append((t.to_sec(), msg.data))

        bag.close()

        # Convert to numpy arrays
        angle_data = np.array(angle_data)
        vgrf_data = np.array(vgrf_data)

        # Interpolate angle data to match vGRF timestamps
        interpolated_angles = np.interp(vgrf_data[:, 0], angle_data[:, 0], angle_data[:, 1])
        interpolated_forces = vgrf_data[:, 1]
        time = vgrf_data[:, 0] - vgrf_data[0, 0]

        # Derivatives of force and angle
        force_derivatives = np.gradient(interpolated_forces)
        angle_derivatives = np.gradient(interpolated_angles)
        
        # Gait phase detector offline
        (gait_phases, gait_progress, start_time, end_time) = self.offline_phase_estimator(time, interpolated_forces)

        #Adjust Data to the Gait Phase anaylized (without the first and last cycle)
        mask = (time >= start_time) & (time <= end_time)
        adjusted_time = time[mask]
        adjusted_force = interpolated_forces[mask]
        adjusted_angle = interpolated_angles[mask]
        adjusted_force_derivatives = force_derivatives[mask]
        adjusted_angle_derivatives = angle_derivatives[mask]
        
        # --- Write data to CSV ---
        rospy.loginfo(f"Saving gait data to {self.labels_path}...")
        with open(self.labels_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Force', 'Force_Derivative', 'Angle', 'Angle_Derivative', 'Gait_Progress', 'Phase'])

            for i in range(len(gait_phases)):
                writer.writerow([
                    adjusted_time[i],  # Time
                    adjusted_force[i],  # Force
                    adjusted_force_derivatives[i],  # Force derivative
                    adjusted_angle[i],  # Angle
                    adjusted_angle_derivatives[i],  # Angle derivative
                    gait_progress[i],  # Gait progress
                    gait_phases[i]  # Phase
                ])

        rospy.loginfo(f"Gait data saved successfully.")

        # Training code here
        y_train = np.array(gait_progress) 
        # Combine features: X (force), z (gait phases), a (angles) and derivatives
        X_combined_train = np.hstack((adjusted_force, gait_phases))
        X_derivative_train = np.diff(adjusted_force, axis=0)  # Calcul des dérivées de la force

        X_combined_train = X_combined_train[:-1]
        X_combined_train = np.hstack((X_combined_train, X_derivative_train))
        X_combined_train = np.hstack((X_combined_train, adjusted_angle.reshape(-1, 1)[:-1]))
        X_combined_train = np.hstack((X_combined_train, adjusted_angle_derivatives.reshape(-1, 1)))
        y_train = y_train[:-2]

        # Standardize the features
        scaler = StandardScaler()
        X_combined_scaled_train = scaler.fit_transform(X_combined_train)

        seq_length = 10
        X_combined_reshaped_train = self.create_sequences(X_combined_scaled_train, seq_length)
        y_seq_train = y_train[seq_length-1:]
        
        # Build the CNN model
        model = Sequential()
        model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_combined_reshaped_train.shape[1], X_combined_reshaped_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(100, activation='softmax'))  # Assuming 4 phases of gait

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        min_length_train = min(len(X_combined_reshaped_train), len(y_seq_train))

        X_combined_reshaped_train = X_combined_reshaped_train[:min_length_train]
        y_seq_train = y_seq_train[:min_length_train]

        # Train the model
        model.fit(X_combined_reshaped_train, y_seq_train, epochs=10, batch_size=32)

        # Final of Training code
        # Load the learning model
        try:
            with open(self.model_path, 'r') as model_file:
                for line in model_file:
                    self.learning_model.append(line.strip().split(','))  # Assuming CSV with commas
            self.modelLoaded = True
            return 1
        except FileNotFoundError:
            rospy.logerr("Model was not loaded, please verify")
            self.modelLoaded = False
            return 0

    def estimate_phase(self):
        if self.modelLoaded and self.ankle_angle is not None and self.ground_force is not None:
            rospy.loginfo(f"Estimating phase for patient {self.patient} using model {self.model_path}...")
            # Estimation code goes here

            # Final Estimation code
            self.ankle_angle = None
            self.ground_force = None

    def run(self):
        if os.path.exists(self.model_path):
            rospy.loginfo(f"Model found for patient {self.patient}. Proceeding with phase estimation.")
            with open(self.model_path, 'r') as model_file:
                for line in model_file:
                    self.learning_model.append(line.strip().split(','))
            self.modelLoaded = True
        else:
            rospy.logwarn(f"Model not found for patient {self.patient}. Training a new model.")
            res = self.train_model()
            if res == 0:
                rospy.signal_shutdown("Model was not found")
        rospy.spin()


if __name__ == '__main__':
    try:
        estimator = GaitPhaseEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
