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

class GaitPhaseEstimator:

    def __init__(self, samples_size=10):
        # ROS setup
        rospy.init_node('gait_phase_estimator', anonymous=True)

        # Paths to models and training data
        self.patient = rospy.get_param("patient", "train")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ankle_exoskeleton')

        self.labels_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_labels1.csv")
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_modelR5.pkl")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"5N1D5KMPH.bag")

        # Variables to track state
        self.modelLoaded = False
        self.model = None
        self.angleUpdated = False
        self.forceUpdated = False

        # Variables for derivatives
        self.last_angle_timestamp = None
        self.last_force_timestamp = None

        # Variable for estimation
        self.samples_size = samples_size  # Taille de la mémoire
        self.current_phase = deque(maxlen=self.samples_size)  # Mémoire circulaire pour les prédictions
        self.ankle_angle = None
        self.ground_force = None
        self.smoothed_estimated_phase = 0
        self.prediction_buffer = []
        self.blocked = False       # État de blocage
        self.block_count = 0       # Compteur de points bloqués
        self.window_size = 0       # Taille de la fenêtre pour débloquer

        # ROS Subscribers and Publishers
        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)
        self.gait_ptg_pub = rospy.Publisher('/gait_percentage_R5N1D5KMPH', Int16, queue_size=2)
        self.force_dt_pub = rospy.Publisher('/ground_force_derivative', Float64, queue_size=2)
        self.angle_dt_pub = rospy.Publisher('/angle_force_derivative', Float64, queue_size=2)
        self.phase_pub = rospy.Publisher('/stance_swing_phase_R5N1D5KMPH.bag', Int16, queue_size=2)  # New publisher for stance/swing phase

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

        model1 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=9, random_state=42)
        model2 = RandomForestRegressor(n_estimators=1000, random_state=42)
        model3 = VotingRegressor(estimators=[
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=1000, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=9, random_state=42))
        ])

        models = {'Gradient Boosting': model1, 'Random Forest': model2, 'Voting Regressor': model3}
        mse_results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_results[name] = mse
            print(f"{name} Mean Squared Error: {mse:.4f}")

        best_model_name = min(mse_results, key=mse_results.get)
        model = models[best_model_name]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        rospy.loginfo(f"Saving model to {self.model_path}...")
        joblib.dump(model, self.model_path)
        rospy.loginfo(f"Model saved successfully.")

        try:
            self.model = joblib.load(self.model_path)
            self.modelLoaded = True
            return 1
        except FileNotFoundError:
            rospy.logerr("Model was not loaded, please verify")
            self.modelLoaded = False
            return 0
    
    def estimate_phase(self, anomaly_threshold=50):
        window_size = self.samples_size
        if not (self.modelLoaded and self.angleUpdated and self.forceUpdated):
            return

        # Étape 1 : Création de la nouvelle prédiction
        current_input = [
            self.ground_force, self.ground_force_derivative,
            self.ankle_angle, self.ankle_angle_derivative
        ]
        try:
            new_phase = float(self.model.predict([current_input])[0])
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return

        # Publier les dérivées
        self.angle_dt_pub.publish(self.ankle_angle_derivative)
        self.force_dt_pub.publish(self.ground_force_derivative)

        # Étape 2 : Gestion du tampon circulaire
        self.prediction_buffer.append(new_phase)
        if len(self.prediction_buffer) > self.samples_size:
            self.prediction_buffer.pop(0)

        # Étape 3 : Traitement et filtrage
        if len(self.prediction_buffer) == self.samples_size:
            predictions = np.clip(np.array(self.prediction_buffer, dtype=np.float32), 0, 100)
            predictions = self.correct_anomalies(predictions, anomaly_threshold)
            smoothed_predictions = self.smooth_predictions(predictions, window_size)
            self.smoothed_estimated_phase = smoothed_predictions[-1]

        # Étape 4 : Publication des résultats
        self.publish_results()

        # Réinitialisation des états
        self.angleUpdated = False
        self.forceUpdated = False

    def correct_anomalies(self, predictions, anomaly_threshold2):
        prediction2_buffer = []

        # Générer une progression linéaire pour `predictions2`
        for j in range(130):
            predictions2 = j * 100 / 130
            prediction2_buffer.append(predictions2)

        # Synchroniser les longueurs de `predictions` et `prediction2_buffer`
        if len(predictions) > len(prediction2_buffer):
            prediction2_buffer.extend([prediction2_buffer[-1]] * (len(predictions) - len(prediction2_buffer)))
        elif len(prediction2_buffer) > len(predictions):
            prediction2_buffer = prediction2_buffer[:len(predictions)]

        for i in range(5, len(predictions)):
            delta_prev = abs(predictions[i] - predictions[i - 1])
            anomaly_threshold = 0.8 * predictions[i - 1]

            # Signal de début de calcul pour prediction2 si prediction est égal à 0
            if predictions[i] == 0:
                print(f"Signal de début de calcul pour prediction2 à l'indice {i}.")
                prediction2_buffer[i] = 0  # Initialisation pour ce cas

            # Détection et correction des anomalies
            if predictions[i - 1] > 70 and delta_prev > anomaly_threshold:
                print(f"Anomalie détectée à l'indice {i}: saut anormal.")
                predictions[i] = 0  # Maintenir la continuité
                self.blocked = True
                self.block_count = 0

                # Appliquer la progression linéaire aux prédictions existantes
                for j in range(len(predictions)):
                    if j < len(prediction2_buffer):
                        if predictions[j] < prediction2_buffer[j]:
                            predictions[j] = prediction2_buffer[j]
                    else:
                        break

                print(f"Blocage levé après {self.block_count} points.")
                self.blocked = False
                self.block_count = 0
                break  # Sortir de la boucle après traitement d'une anomalie

            # Gestion de l'état bloqué
            if self.blocked:
                self.block_count += 1
                for j in range(len(predictions)):
                    if j < len(prediction2_buffer):
                        if predictions[j] < prediction2_buffer[j]:
                            predictions[j] = prediction2_buffer[j]
                    else:
                        break

                #print(f"Blocage levé après {self.block_count} points.")
                self.blocked = False
                self.block_count = 0

            # Comparer `predictions` et `prediction2_buffer` à l'indice courant
            if i < len(prediction2_buffer) and predictions[i] > prediction2_buffer[i]:
                # print(f"Correction basée sur predictions2 à l'indice {i}.")
                predictions[i] = prediction2_buffer[i]

            # Correction des régressions et des anomalies
            if predictions[i] < predictions[i - 1] or predictions[i] - predictions[i - 1] > 2 * (predictions[i - 1] - predictions[i - 2]):
                #print(f"Correction à l'indice {i}: régression détectée.")
                predictions[i] = predictions[i - 1] + (predictions[i - 1] - predictions[i - 2])
                predictions[i] = statistics.mean(predictions)

            if predictions[i - 1] == 0 and predictions[i] > 5:
                #print(f"Pic détecté après un zéro à l'indice {i}. Correction appliquée.")
                predictions[i] = 2

        return predictions

    def smooth_predictions(self, predictions, window_size):
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(predictions, kernel, mode='same')
        correction_factor = window_size / (window_size - (window_size // 2))
        smoothed *= correction_factor
        smoothed = smoothed - 4

        return np.clip(smoothed, 0, 100)

    def publish_results(self):
        self.gait_ptg_pub.publish(int(self.smoothed_estimated_phase))

        phase_indicator = Int16()
        phase_indicator.data = 100 if self.ground_force == 0 else 0
        self.phase_pub.publish(phase_indicator)

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
