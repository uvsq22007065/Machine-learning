#!/usr/bin/env python3

import rospy, rospkg, rosbag
from std_msgs.msg import Float64, Int16
from moticon_insole.msg import InsoleData
import os, logging
import time as times
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Remove warnings for GPU (TODO: Review if it affect the algorithm)
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from scipy.signal import butter, filtfilt 
import matplotlib.pyplot as plt
from joblib import dump, load
from collections import deque

class GaitPhaseEstimator:
    def __init__(self, samples_size=10):
        # ROS setup
        rospy.init_node('gait_phase_estimator_LSTM', anonymous=True)

        # Paths to models and training data
        self.patient = rospy.get_param("patient", "subject1")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('gait_vector_estimator')

        self.labels_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_labelsLSTM.xlsx")
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_modelLSTM.pkl")
        self.scaler_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_modelLSTM_scaler.pkl")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"{self.patient}/{self.patient}_train_merged.bag")
        self.log_file_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_modelLSTM_log.txt")

        # Sampling frequency
        self.fs = 100
        
        # Butterworth filters for angle, velocity, force, and CoP
        fc = 5  # cutoff frequency for angle, force, and CoP
        self.b, self.a = butter(3, fc / (self.fs / 2), btype='low')     
        fc_vel = 10  # cutoff frequency for velocity
        self.b_vel, self.a_vel = butter(3, fc_vel / (self.fs / 2), btype='low')

        # Buffers
        self.window_size = 20
        self.angles, self.forces, self.cop_x = [], [], []
        self.ankle_angle_derivative = 0
        self.ankle_angle = 0

        # Variables to track state
        self.modelLoaded = False
        self.model = []
        self.angleUpdated = False
        self.forceUpdated = False

        self.force_threshold = 0.04

        #Variable for estimation
        self.samples_size = samples_size
        self.data_sequence = deque(maxlen=self.samples_size)
        self.ankle_angle = None
        self.ground_force = None
        # self.smoothed_estimated_phase = 0
        self.cop = None
        self.prev_phase = 0
        self.prediction_buffer = []

    def setup_logger(self):
        log_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setFormatter(log_formatter)

        roslog = logging.getLogger('rosout')
        roslog.setLevel(logging.INFO)
        roslog.addHandler(file_handler)

    def sub_pub_initialization(self):
        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/moticon_insole_data/left', InsoleData, self.ground_force_callback)
        self.gait_ptg_pub = rospy.Publisher('/gait_percentage_LSTM', Int16, queue_size=2)
        self.force_dt_pub = rospy.Publisher('/force_derivative_LSTM', Float64, queue_size=2)
        self.angle_dt_pub = rospy.Publisher('/angle_derivative_LSTM', Float64, queue_size=2)
        self.phase_pub = rospy.Publisher('/stance_swing_phase_LSTM', Int16, queue_size=2)  
        
    def ankle_angle_callback(self, msg):
        """Updates Angle Buffer (IMU)"""
        self.angles.append(msg.data)

        if len(self.angles) > self.window_size:
            angle_rad = np.deg2rad(self.angles)
            angle_filtered = filtfilt(self.b, self.a, angle_rad)

            vel_rps = np.diff(angle_filtered) #* self.fs
            # vel_filtered = lfilter(self.b_vel, self.a_vel, vel_rps) 

            self.ankle_angle = angle_filtered[-1]
            self.ankle_angle_derivative = vel_rps[-1]
            self.angleUpdated = True

            self.angles.pop(0)

    def ground_force_callback(self, msg):
        """Updates Force Buffer (Insole)"""
        self.forces.append(msg.normalised_force if msg.normalised_force >= self.force_threshold else 0.0)
        self.cop_x.append(msg.cop_x)

        if len(self.forces) > self.window_size:
            force_filtered = filtfilt(self.b, self.a, self.forces)
            force_filtered = np.array([f if f >= self.force_threshold else 0.0 for f in force_filtered])

            force_derivative = np.diff(force_filtered)
            # force_derivative_filtered = filtfilt(self.b, self.a, force_derivative)

            self.ground_force = force_filtered[-1]
            self.ground_force_derivative = force_derivative[-1]
            self.cop = self.cop_x[-1]
            self.forceUpdated = True

            self.forces.pop(0)

    
    def offline_phase_estimator(self, time, interpolated_forces_raw):
        stance_mask = interpolated_forces_raw >= self.force_threshold  # True for stance, False for swing
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

        return gait_phases, gait_progress, start_time, end_time, stance_mask
    
    def select_consistent_cycles(self, cycles_angle, cycles_phase, percentage=70):
        # Creation of a common phase standardised from 0 to 100% with 1000 points
        common_phase = np.linspace(0, 100, 1000)
        interpolated_angles = []  # List of interpolated angles on the common phase
        valid_indices = []        # Indices of valid cycles

        # Loop on all cycles supplied
        for i, (angle, phase) in enumerate(zip(cycles_angle, cycles_phase)):
            if len(phase) < 2:
                continue  # Cycles too short to interpolate are ignored

            # Interpolation of the angle on the common phase
            interp = np.interp(common_phase, phase, angle, left=np.nan, right=np.nan)

            # Cycles whose interpolation does not contain NaN
            if not np.isnan(interp).any():
                interpolated_angles.append(interp)
                valid_indices.append(i)

        interpolated_angles = np.array(interpolated_angles)

        # If fewer than 3 valid cycles, cannot calculate a reliable average.
        if len(interpolated_angles) < 3:
            return valid_indices

        # Calculating the average cycle for comparison
        mean_angle = np.mean(interpolated_angles, axis=0)

        # Calculation of the similarity score (correlation) of each cycle with the average cycle
        similarity_scores = np.array([
            np.corrcoef(row, mean_angle)[0, 1] for row in interpolated_angles
        ])

        # Sort indices by decreasing similarity
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Selection of a given percentage of the best cycles
        n_select = max(1, round((percentage / 100) * len(sorted_indices)))
        selected_indices = [valid_indices[i] for i in sorted_indices[:n_select]]

        return selected_indices  # Returns the indices of the most consistent cycles


    def create_dataset_per_cycles(self, adjusted_force, adjusted_force_derivatives, adjusted_angle,
                                adjusted_angle_derivatives, adjusted_cop, adjusted_time, gait_phases, gait_progress):
        # Lists for storing extracted cycles
        cycles_force = []
        cycles_force_deriv = []
        cycles_angle = []
        cycles_angle_deriv = []
        cycles_cop = []
        cycles_time = []
        cycles_phase = []
        cycles_progress = []

        # Temporary dictionary for accumulating cycle data
        current_cycle = {'force': [], 'force_d': [], 'angle': [], 'angle_d': [],'cop': [], 'time': [], 'phase': [], 'progress': []}

        # Loop on each data point
        for i, label in enumerate(gait_phases):
            # Filling the current cycle
            current_cycle['force'].append(adjusted_force[i])
            current_cycle['force_d'].append(adjusted_force_derivatives[i])
            current_cycle['angle'].append(adjusted_angle[i])
            current_cycle['angle_d'].append(adjusted_angle_derivatives[i])
            current_cycle['cop'].append(adjusted_cop[i])
            current_cycle['time'].append(adjusted_time[i])
            current_cycle['phase'].append(label)
            current_cycle['progress'].append(gait_progress[i])

            # End of cycle detected: swing -> stance transition or end of data
            if label == 'swing_phase' and (i + 1 == len(gait_phases) or gait_phases[i + 1] == 'stance_phase'):
                # Cycle storage complete
                cycles_force.append(np.array(current_cycle['force']))
                cycles_force_deriv.append(np.array(current_cycle['force_d']))
                cycles_angle.append(np.array(current_cycle['angle']))
                cycles_angle_deriv.append(np.array(current_cycle['angle_d']))
                cycles_cop.append(np.array(current_cycle['cop']))
                cycles_time.append(np.array(current_cycle['time']))
                # Creation of a linear phase from 0 to 100% for the cycle
                cycles_phase.append(np.linspace(0, 100, len(current_cycle['angle'])))
                cycles_progress.append(np.array(current_cycle['progress']))

                # Reset for next cycle
                current_cycle = {'force': [], 'force_d': [], 'angle': [], 'angle_d': [], 'cop': [],'time': [], 'phase': [], 'progress': []}

        # If no cycle has been identified, return the original data
        if len(cycles_angle) == 0:
            return adjusted_force, adjusted_force_derivatives, adjusted_angle, adjusted_angle_derivatives, adjusted_cop, adjusted_time, gait_phases, gait_progress

        # Selection of the most coherent cycles
        selected_indices_angle = self.select_consistent_cycles(cycles_angle, cycles_phase, percentage=90)
        selected_indices_force = self.select_consistent_cycles(cycles_force, cycles_phase, percentage=90)
        selected_indices = list(set(selected_indices_angle) & set(selected_indices_force))

        # Reconstruction of filtered signals from selected cycles
        filtered_force = np.concatenate([cycles_force[i] for i in selected_indices])
        filtered_force_d = np.concatenate([cycles_force_deriv[i] for i in selected_indices])
        filtered_angle = np.concatenate([cycles_angle[i] for i in selected_indices])
        filtered_angle_d = np.concatenate([cycles_angle_deriv[i] for i in selected_indices])
        filtered_cop = np.concatenate([cycles_cop[i] for i in selected_indices])
        filtered_time = np.concatenate([cycles_time[i] for i in selected_indices])

        # Phase reconstruction: if < 60%, consider stance, otherwise swing
        filtered_phase = np.concatenate([['stance_phase' if p < 60 else 'swing_phase' for p in cycles_phase[i]] for i in selected_indices])
        filtered_progress = np.concatenate([cycles_progress[i] for i in selected_indices])

        ptg_data = (len(filtered_time)*100.0/len(adjusted_time))
        rospy.loginfo("Percentage of data used to train: " + str(ptg_data))

        # Returns filtered signals
        return filtered_force, filtered_force_d, filtered_angle, filtered_angle_d, filtered_cop, filtered_time, filtered_phase, filtered_progress 
        
    def train_model(self):
        rospy.loginfo(f"Training model for patient {self.patient}...")
        self.setup_logger()

        # Check if the bag file exists
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"No .bag file found for patient {self.patient} in {self.bag_path}. Training cannot proceed.")
            rospy.signal_shutdown("Missing .bag file for training.")
            return

        # Read bag file and extract data
        bag = rosbag.Bag(self.bag_path)
        angle_data = []
        vgrf_data = []
        cop_data = []

        for topic, msg, t in bag.read_messages(topics=['/ankle_joint/angle', '/moticon_insole_data/left']):
            if topic == '/ankle_joint/angle':
                angle_data.append((t.to_sec(), msg.data))
            elif topic == '/moticon_insole_data/left':
                force_value = msg.normalised_force if msg.normalised_force >= self.force_threshold else 0.0
                vgrf_data.append((t.to_sec(), force_value))
                cop_data.append(msg.cop_x)

        bag.close()
        angle_data = np.array(angle_data)
        vgrf_data = np.array(vgrf_data)
        cop_data = np.array(cop_data)

        interpolated_angles = np.interp(vgrf_data[:, 0], angle_data[:, 0], angle_data[:, 1])
        interpolated_forces = vgrf_data[:, 1]
        time = vgrf_data[:, 0] - vgrf_data[0, 0]

        ''' Angle Filter '''
        angle_rad = np.deg2rad(interpolated_angles)
        angle_filtered = filtfilt(self.b, self.a, angle_rad)

        vel_rps = np.diff(angle_filtered) #* self.fs
        vel_rps = np.append(0, vel_rps)
        # vel_filtered = filtfilt(self.b_vel, self.a_vel, vel_rps)
        
        angles = angle_filtered
        angles_derivative = vel_rps

        ''' Ground Force Filter '''
        # Force filter and correction
        force_filtered = filtfilt(self.b, self.a, interpolated_forces)
        force_filtered = np.array([f if f >= self.force_threshold else 0.0 for f in interpolated_forces])

        # Derivative of force
        f_derivative = np.diff(force_filtered)
        f_derivative = np.append(0,f_derivative)
        # force_derivative_filtered = filtfilt(self.b, self.a, force_derivative)

        forces = force_filtered
        forces_derivative = f_derivative

        cop = cop_data
        
        ''' Gait Phase Estimator offline '''
        (gait_phases, gait_progress, start_time, end_time, stance_mask) = self.offline_phase_estimator(time, interpolated_forces)

        # force_filtered[stance_mask == 0] = 0
        interpolated_forces[stance_mask == 0] = 0

        mask = (time >= start_time) & (time <= end_time)
        adjusted_time = time[mask]
        adjusted_force = forces[mask]
        adjusted_angle = angles[mask]
        adjusted_force_derivatives = forces_derivative[mask]
        adjusted_angle_derivatives = angles_derivative[mask]
        adjusted_cop = cop[mask]
        
        rospy.loginfo(f"Saving gait data to {self.labels_path}...")

        ''' Select most consistent trials '''
        [filtered_force, filtered_force_d, filtered_angle, filtered_angle_d, filtered_cop, filtered_time, filtered_phase, filtered_progress] = self.create_dataset_per_cycles(adjusted_force, adjusted_force_derivatives, adjusted_angle,
                            adjusted_angle_derivatives, adjusted_cop, adjusted_time, gait_phases, gait_progress) 

        df1 = pd.DataFrame({
            'Time': adjusted_time,
            'Force': adjusted_force,
            'Force_Derivative': adjusted_force_derivatives,
            'Angle': adjusted_angle,
            'Angle_Derivative': adjusted_angle_derivatives,
            'CoP': adjusted_cop,
            'Gait_Progress': gait_progress,
            'Phase': gait_phases
        })

        df2 = pd.DataFrame({
            'Time': filtered_time,
            'Force': filtered_force,
            'Force_Derivative': filtered_force_d,
            'Angle': filtered_angle,
            'Angle_Derivative': filtered_angle_d,
            'CoP': filtered_cop,
            'Gait_Progress': filtered_progress,
            'Phase': filtered_phase
        })

        # Save to different sheets in the same file
        with pd.ExcelWriter(self.labels_path, engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='Raw Data', index=False)
            df2.to_excel(writer, sheet_name='Filtered Data', index=False)

        rospy.loginfo(f"Gait data saved successfully.")

        X = np.column_stack((filtered_force, filtered_force_d, filtered_angle, filtered_angle_d, filtered_cop))
        y = np.array(filtered_progress)

        # Data Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dump(scaler, self.scaler_path)
        rospy.loginfo(f"Scaler saved to {self.scaler_path}")

        X = X_scaled.reshape((X.shape[0], 1, X.shape[1]))

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo CNN + LSTM
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))  # First LSTM layer
        model.add(Dropout(0.2))  # Regularization to prevent overfitting
        model.add(LSTM(50, activation='relu', return_sequences=False))  # Second LSTM layer with return_sequences=False
        model.add(Dropout(0.2))  # Another Dropout layer
        model.add(Dense(1))  # Output layer for regression

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Training
        rospy.loginfo("Training the LSTM model...")

        initial_time = times.time()

        history = model.fit(
            X_train, y_train,
            epochs=1000,
            batch_size=32,
            verbose=0,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        final_time = times.time()

        # Calculate and print elapsed time
        training_duration = final_time - initial_time
        rospy.loginfo(f"Training time: {training_duration:.2f} seconds")

        # Loss History
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Model Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rospy.loginfo(f"Performance Metrics:\nMSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        rospy.loginfo("Last Train and Validation Loss for CNN+LSTM Model:")
        rospy.loginfo(f"Train Loss={train_loss[-1]:.4f}, Validation Loss={val_loss[-1]:.4f}")

        # Graficar pérdidas
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('LSTM Training & Validation Loss')
        plt.legend()
        loss_plot_path = os.path.join(os.path.dirname(self.model_path), f"{self.patient}_lstm_loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()

        # Save Model
        rospy.loginfo(f"Saving LSTM model to {self.model_path}...")
        model.save(self.model_path)
        rospy.loginfo(f"Model saved successfully.")


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
        if self.modelLoaded and self.angleUpdated and self.forceUpdated:
            # Créer une nouvelle prédiction basée sur les entrées actuelles
            current_input = [
                self.ground_force, self.ground_force_derivative,
                self.ankle_angle, self.ankle_angle_derivative,
                self.cop
            ]
        
            # Reshape input to match the expected shape of the model (None, 1, 4)
            current_input = np.array(current_input).reshape(1, 1, 5)
        
            # Make the prediction
            new_phase = float(self.model.predict(current_input, verbose=0)[0])  # Check for the last element

            # Publier les dérivées
            self.angle_dt_pub.publish(self.ankle_angle_derivative)
            self.force_dt_pub.publish(self.ground_force_derivative)
            # self.prediction_buffer.append(new_phase)
            # if len(self.prediction_buffer) > self.samples_size:
            #     self.prediction_buffer.pop(0)
            #     smoothed_predictions = self.mean_filter(self.prediction_buffer, self.samples_size)
            #     self.smoothed_estimated_phase = smoothed_predictions[-1]


            # Publish results
            self.gait_ptg_pub.publish(int(new_phase))


            # Publier l'indicateur de phase (swing ou stance)
            phase_indicator = Int16()
            phase_indicator.data = 100 if self.ground_force == 0 else 0
            self.phase_pub.publish(phase_indicator)

            # Réinitialiser les indicateurs d'angle et de force
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
        
        self.sub_pub_initialization()
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

