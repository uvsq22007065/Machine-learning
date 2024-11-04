#!/usr/bin/env python3

import rospy, rospkg, rosbag
from std_msgs.msg import Float64, Int16
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Remove warnings for GPU (TODO: Review if it affect the algorithm)
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from collections import deque

class GaitPhaseEstimator:
    def __init__(self):
        # ROS setup
        rospy.init_node('gait_phase_estimator', anonymous=True)

        # Paths to models and training data
        self.patient = rospy.get_param("patient", "test")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ankle_exoskeleton')

        self.labels_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_labels.csv")
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_model.keras")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"{self.patient}.bag")

        # Variables to track state
        self.modelLoaded = False
        self.model = []
        self.angleUpdated = False
        self.forceUpdated = False

        # Variables for derivatives
        self.last_angle_timestamp = None
        self.last_force_timestamp = None

        #Variable for estimation
        self.samples_size = 130
        self.data_sequence = deque(maxlen=self.samples_size)
        self.ankle_angle = None
        self.ground_force = None

        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)
        self.gait_ptg_pub = rospy.Publisher('/gait_percentage', Int16, queue_size=2)
        self.force_dt_pub = rospy.Publisher('/ground_force_derivative', Float64, queue_size=2)
        self.angle_dt_pub = rospy.Publisher('/angle_force_derivative', Float64, queue_size=2)


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

        #gait_progress = np.round(gait_progress / 2) * 2
        #gait_progress[gait_progress > 98] = 0

        return gait_phases, gait_progress, start_time, end_time
        
    # Fenêtrage des données pour Conv1D
    def create_sequences(self, data, labels, seq_length):
        sequences, label_sequences = [], []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
            label_sequences.append(labels[i + seq_length - 1])
        return np.array(sequences), np.array(label_sequences)

    # Concaténation des caractéristiques
    def prepare_features(self, X, X_Derivative, a, a_derivative):
        features = np.hstack((X.reshape(-1, 1), 
                              X_Derivative.reshape(-1, 1), 
                              a.reshape(-1, 1), 
                              a_derivative.reshape(-1, 1)))
        return features


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

        # Conversion des données en tableaux numpy
        X_train = np.array(adjusted_force).reshape(-1, 1)
        y_train = np.array(gait_progress).flatten()
        X_train_Derivative = np.array(adjusted_force_derivatives).flatten()
        a_train = np.array(adjusted_angle).flatten()
        a_train_Derivative = np.array(adjusted_angle_derivatives).flatten()

        # Training code
        X_combined_train = self.prepare_features(X_train, X_train_Derivative, a_train, a_train_Derivative)
        y_train = np.array(gait_progress)
        #print("X_combined_train shape:", X_combined_train.shape)

        # Standardize the features
        scaler = StandardScaler()
        X_combined_scaled_train = scaler.fit_transform(X_combined_train)

        seq_length = self.samples_size
        X_seq_train, y_seq_train = self.create_sequences(X_combined_scaled_train, y_train, seq_length)
        y_seq_train = y_train[seq_length-1:]
        print("X_combined_reshaped_train shape:", X_seq_train.shape)

        # Build the CNN model
        model = Sequential([
            Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=False),  # Ajout d'une couche LSTM
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])

        # Compiler le modèle avec un learning rate plus bas et une fonction de perte adaptée
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

        # Callback pour l'early stopping
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Ajustement du modèle avec plus d'epochs et le callback d'early stopping
        model.fit(
            X_seq_train, y_seq_train, 
            epochs=10,  # Augmentation du nombre d'epochs
            batch_size=32,
            callbacks=[early_stopping]  # Ajout de l'early stopping
        )

        # Entraînement du modèle avec validation croisée
        model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32, callbacks=[early_stopping], verbose=1)

        # --- Save model in HDF5 format---
        rospy.loginfo(f"Saving model to {self.model_path}...")
        model.save(self.model_path)   
        rospy.loginfo(f"Model saved successfully.")
        # Final of Training code

        # Load the learning model
        try:
            self.model = load_model(self.model_path)
            self.modelLoaded = True
            return 1
        except FileNotFoundError:
            rospy.logerr("Model was not loaded, please verify")
            self.modelLoaded = False
            return 0

    def estimate_phase(self):
        if ((self.modelLoaded is not None) and (self.angleUpdated == True) and (self.forceUpdated == True)):
            # Estimation code 
            current_input = [self.ground_force, self.ground_force_derivative, self.ankle_angle, self.ankle_angle_derivative]

            # FOR TEST
            self.angle_dt_pub.publish(self.ankle_angle_derivative)
            self.force_dt_pub.publish(self.ground_force_derivative)
            #print(self.ground_force_derivative)

            # Add the current input to the deque, which maintains the last 10 timesteps
            self.data_sequence.append(current_input)

            if len(self.data_sequence) == self.samples_size:
                # Convert deque to numpy array and reshape for model input
                x_combined_data = np.array(self.data_sequence).reshape(1, self.samples_size, 4)

                # Perform prediction
                estimated_phase = self.model.predict(x_combined_data, verbose=0).flatten()
                self.gait_ptg_pub.publish(int(estimated_phase))
                #self.data_sequence = []
            
            # Final Estimation code
            self.angleUpdated = False
            self.forceUpdated = False
    
    def run(self):
        rate = rospy.Rate(200)  # Set the frequency to 200 Hz

        if os.path.exists(self.model_path):
            rospy.loginfo(f"Model found for patient {self.patient}. Proceeding with phase estimation.")
            self.model = load_model(self.model_path)
            self.modelLoaded = True
        else:
            rospy.logwarn(f"Model not found for patient {self.patient}. Training a new model.")
            res = self.train_model()
            if res == 0:
                rospy.signal_shutdown("Model was not found")
        
        # Main loop running at 200 Hz
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