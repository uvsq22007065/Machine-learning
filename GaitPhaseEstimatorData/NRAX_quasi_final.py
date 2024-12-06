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
import statistics

class GaitPhaseEstimator:
    def __init__(self):
        # ROS setup
        rospy.init_node('gait_phase_estimator', anonymous=True)

        # Paths to models and training data
        self.patient = rospy.get_param("patient", "daniel_train")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ankle_exoskeleton')

        self.labels_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_labels.csv")
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_modelNRAX.keras")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"{self.patient}.bag")

        # Variables to track state
        self.modelLoaded = False
        self.model = []
        self.angleUpdated = False
        self.forceUpdated = False

        # Variables for derivatives
        self.last_angle_timestamp = None
        self.last_force_timestamp = None

        # Variable for estimation
        self.samples_size = 12
        self.data_sequence = deque(maxlen=self.samples_size)
        self.ankle_angle = None
        self.ground_force = None
        self.smoothed_estimated_phase = 0
        self.prediction_buffer = []
        self.smoothed_estimated_phase = 0

        # ROS Subscribers and Publishers
        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)
        self.gait_ptg_pub = rospy.Publisher('/gait_percentage_NRAX', Int16, queue_size=2)
        self.force_dt_pub = rospy.Publisher('/ground_force_derivative', Float64, queue_size=2)
        self.angle_dt_pub = rospy.Publisher('/angle_force_derivative', Float64, queue_size=2)
        self.phase_pub = rospy.Publisher('/stance_swing_phase_NRAX', Int16, queue_size=2)  # New publisher for stance/swing phase
        self.current_phase = []

    def build_nrax_model(self, input_shape):
        """Create a NRAX model with an attention mechanism."""
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(64, activation='relu')(inputs)
        attention_output = layers.Attention()([x, x])
        x = layers.Concatenate()([x, attention_output])
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1)(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

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
        
    def train_model(self):
        rospy.loginfo(f"Training model for patient {self.patient}...")
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"No .bag file found for patient {self.patient} in {self.bag_path}. Training cannot proceed.")
            rospy.signal_shutdown("Missing .bag file for training.")
            return

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        input_shape = (X_train.shape[1],)
        self.model = self.build_nrax_model(input_shape)

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        # Fit the model
        self.model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1, callbacks=[early_stopping])

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        rospy.loginfo(f"Saving self.model to {self.model_path}...")
        joblib.dump(self.model, self.model_path)
        rospy.loginfo(f"Model saved successfully.")

        try:
            self.model = joblib.load(self.model_path)
            self.modelLoaded = True
            return 1
        except FileNotFoundError:
            rospy.logerr("Model was not loaded, please verify")
            self.modelLoaded = False
            return 0

    def mean_filter(self, predictions, window_size, anomaly_threshold=50):
        predictions = np.clip(np.array(predictions, dtype=np.float32), 0, 100)

        # Détection des anomalies et régressions AVANT lissage
        for i in range(5, len(predictions)):
            delta_prev = abs(predictions[i] - predictions[i - 1])

            if delta_prev > anomaly_threshold:
                print(f"Anomalie détectée à l'indice {i}: saut anormal.")
                predictions = [0]
                break
            elif (predictions[i] < predictions[i - 1] or
                  predictions[i] - predictions[i - 1] > 2 * predictions[i - 1] - predictions[i - 2]):
                print(f"Correction à l'indice {i}: régression détectée.")
                predictions[i] = predictions[i - 1]
                predictions[i] = statistics.mean(predictions)

        # Lissage par moyenne glissante
        smoothed_predictions = np.convolve(predictions, np.ones(window_size) / window_size, mode='same')

        # Correction pour atténuation
        correction_factor = window_size / (window_size - (window_size // 2))
        smoothed_predictions *= correction_factor

        if smoothed_predictions[-1] > 100:
            smoothed_predictions[-1] = 100
        if smoothed_predictions[-1] < 0:
            smoothed_predictions[-1] = 0

        return smoothed_predictions

    def estimate_phase(self):
        if self.modelLoaded and self.angleUpdated and self.forceUpdated:
            # Créer une nouvelle prédiction basée sur les entrées actuelles
            current_input = [
                self.ground_force, self.ground_force_derivative,
                self.ankle_angle, self.ankle_angle_derivative
            ]
        
            # Reshape input to match the expected shape of the model (None, 1, 4)
            current_input = np.array(current_input).reshape(1, 4)
        
            # Make the prediction
            new_phase = float(self.model.predict(current_input)[0])

            # Publier les dérivées
            self.angle_dt_pub.publish(self.ankle_angle_derivative)
            self.force_dt_pub.publish(self.ground_force_derivative)
            self.prediction_buffer.append(new_phase)
            if len(self.prediction_buffer) > self.samples_size:
                self.prediction_buffer.pop(0)
                smoothed_predictions = self.mean_filter(self.prediction_buffer, self.samples_size)
                self.smoothed_estimated_phase = smoothed_predictions[-1]

            # Publier les résultats
            self.gait_ptg_pub.publish(int(self.smoothed_estimated_phase))

            # Publier l'indicateur de phase (swing ou stance)
            phase_indicator = Int16()
            phase_indicator.data = 100 if self.ground_force == 0 else 0
            self.phase_pub.publish(phase_indicator)

            # Réinitialiser les indicateurs d'angle et de force
            self.angleUpdated = False
            self.forceUpdated = False

    def run(self):
        rate = rospy.Rate(200)
        if os.path.exists(self.model_path):
            rospy.loginfo(f"Model found for patient {self.patient}. Proceeding with phase estimation.")
            self.model = joblib.load(self.model_path)
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
