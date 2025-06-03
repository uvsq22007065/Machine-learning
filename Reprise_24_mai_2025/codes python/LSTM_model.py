#!/usr/bin/env python3

import os
import logging
import time as times
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Remove warnings for GPU
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
from datetime import datetime


class GaitPhaseEstimator:
    def __init__(self, data_folder, patient_id="subject8", samples_size=10):
        # Setup paths
        self.data_folder = data_folder  # Ajoutez cette ligne pour définir l'attribut data_folder
        self.patient = patient_id
        self.base_path = os.path.abspath(data_folder)  # Convert to absolute path
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)  # Added exist_ok=True
        self.labels_path = os.path.join(self.current_results_folder, f"{self.patient}_labelsCNN.csv")
        self.model_path = os.path.join(self.current_results_folder, f"{self.patient}_modelLSTM.keras")
        self.scaler_path = os.path.join(self.current_results_folder, f"{self.patient}_modelLSTM_scaler.pkl")
        self.log_file_path = os.path.join(self.current_results_folder, f"{self.patient}_modelLSTM_log.txt")

        # Initialize parameters
        self.fs = 100
        self.force_threshold = 0.04
        self.sequence_length = 130
        self.samples_size = samples_size
        
        # Setup filters
        fc = 5
        self.b, self.a = butter(3, fc / (self.fs / 2), btype='low')
        fc_vel = 10
        self.b_vel, self.a_vel = butter(3, fc_vel / (self.fs / 2), btype='low')

        # Initialize model state
        self.modelLoaded = False
        self.model = None
        self.scaler = None
        
        # Setup logging
        self.setup_logger()

        # Créer le dossier results s'il n'existe pas
        self.results_folder = os.path.join(os.path.dirname(data_folder), "results")
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        # Créer un sous-dossier avec la date et l'heure
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_results_folder = os.path.join(self.results_folder, f"LSTM_{patient_id}_final_training_{now}")
        os.makedirs(self.current_results_folder)
            
    def setup_logger(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(self.log_file_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def update_angle_data(self, angle_deg):
        """Updates Angle Buffer (equivalent to IMU callback)"""
        self.angles.append(angle_deg)

        if len(self.angles) > self.window_size:
            angle_rad = np.deg2rad(self.angles)
            angle_filtered = filtfilt(self.b, self.a, angle_rad)

            vel_rps = np.diff(angle_filtered)
            
            self.ankle_angle = angle_filtered[-1]
            self.ankle_angle_derivative = vel_rps[-1] if len(vel_rps) > 0 else 0
            self.angleUpdated = True

            self.angles.pop(0)

    def update_force_data(self, normalized_force, cop_x):
        """Updates Force Buffer (equivalent to insole callback)"""
        force_value = normalized_force if normalized_force >= self.force_threshold else 0.0
        self.forces.append(force_value)
        self.cop_x.append(cop_x)

        if len(self.forces) > self.window_size:
            force_filtered = filtfilt(self.b, self.a, self.forces)
            force_filtered = np.array([f if f >= self.force_threshold else 0.0 for f in force_filtered])

            force_derivative = np.diff(force_filtered)

            self.ground_force = force_filtered[-1]
            self.ground_force_derivative = force_derivative[-1] if len(force_derivative) > 0 else 0
            self.cop = self.cop_x[-1]
            self.forceUpdated = True

            self.forces.pop(0)
            self.cop_x.pop(0)

    def offline_phase_estimator(self, time, interpolated_forces_raw):
        """Offline gait phase estimation"""
        stance_mask = interpolated_forces_raw >= self.force_threshold
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
        """Select most consistent gait cycles"""
        common_phase = np.linspace(0, 100, 1000)
        interpolated_angles = []
        valid_indices = []

        for i, (angle, phase) in enumerate(zip(cycles_angle, cycles_phase)):
            if len(phase) < 2:
                continue

            interp = np.interp(common_phase, phase, angle, left=np.nan, right=np.nan)

            if not np.isnan(interp).any():
                interpolated_angles.append(interp)
                valid_indices.append(i)

        interpolated_angles = np.array(interpolated_angles)

        if len(interpolated_angles) < 3:
            return valid_indices

        mean_angle = np.mean(interpolated_angles, axis=0)

        similarity_scores = np.array([
            np.corrcoef(row, mean_angle)[0, 1] for row in interpolated_angles
        ])

        sorted_indices = np.argsort(similarity_scores)[::-1]
        n_select = max(1, round((percentage / 100) * len(sorted_indices)))
        selected_indices = [valid_indices[i] for i in sorted_indices[:n_select]]

        return selected_indices

    def create_dataset_per_cycles(self, adjusted_force, adjusted_force_derivatives, adjusted_angle,
                                adjusted_angle_derivatives, adjusted_cop, adjusted_time, gait_phases, gait_progress):
        """Create dataset organized by gait cycles"""
        cycles_force = []
        cycles_force_deriv = []
        cycles_angle = []
        cycles_angle_deriv = []
        cycles_cop = []
        cycles_time = []
        cycles_phase = []
        cycles_progress = []

        current_cycle = {'force': [], 'force_d': [], 'angle': [], 'angle_d': [],'cop': [], 'time': [], 'phase': [], 'progress': []}

        for i, label in enumerate(gait_phases):
            current_cycle['force'].append(adjusted_force[i])
            current_cycle['force_d'].append(adjusted_force_derivatives[i])
            current_cycle['angle'].append(adjusted_angle[i])
            current_cycle['angle_d'].append(adjusted_angle_derivatives[i])
            current_cycle['cop'].append(adjusted_cop[i])
            current_cycle['time'].append(adjusted_time[i])
            current_cycle['phase'].append(label)
            current_cycle['progress'].append(gait_progress[i])

            if label == 'swing_phase' and (i + 1 == len(gait_phases) or gait_phases[i + 1] == 'stance_phase'):
                cycles_force.append(np.array(current_cycle['force']))
                cycles_force_deriv.append(np.array(current_cycle['force_d']))
                cycles_angle.append(np.array(current_cycle['angle']))
                cycles_angle_deriv.append(np.array(current_cycle['angle_d']))
                cycles_cop.append(np.array(current_cycle['cop']))
                cycles_time.append(np.array(current_cycle['time']))
                cycles_phase.append(np.linspace(0, 100, len(current_cycle['angle'])))
                cycles_progress.append(np.array(current_cycle['progress']))

                current_cycle = {'force': [], 'force_d': [], 'angle': [], 'angle_d': [], 'cop': [],'time': [], 'phase': [], 'progress': []}

        if len(cycles_angle) == 0:
            return adjusted_force, adjusted_force_derivatives, adjusted_angle, adjusted_angle_derivatives, adjusted_cop, adjusted_time, gait_phases, gait_progress

        selected_indices_angle = self.select_consistent_cycles(cycles_angle, cycles_phase, percentage=90)
        selected_indices_force = self.select_consistent_cycles(cycles_force, cycles_phase, percentage=90)
        selected_indices = list(set(selected_indices_angle) & set(selected_indices_force))

        filtered_force = np.concatenate([cycles_force[i] for i in selected_indices])
        filtered_force_d = np.concatenate([cycles_force_deriv[i] for i in selected_indices])
        filtered_angle = np.concatenate([cycles_angle[i] for i in selected_indices])
        filtered_angle_d = np.concatenate([cycles_angle_deriv[i] for i in selected_indices])
        filtered_cop = np.concatenate([cycles_cop[i] for i in selected_indices])
        filtered_time = np.concatenate([cycles_time[i] for i in selected_indices])

        filtered_phase = np.concatenate([['stance_phase' if p < 60 else 'swing_phase' for p in cycles_phase[i]] for i in selected_indices])
        filtered_progress = np.concatenate([cycles_progress[i] for i in selected_indices])

        ptg_data = (len(filtered_time)*100.0/len(adjusted_time))
        self.logger.info(f"Percentage of data used to train: {ptg_data:.2f}%")

        return filtered_force, filtered_force_d, filtered_angle, filtered_angle_d, filtered_cop, filtered_time, filtered_phase, filtered_progress 

    def load_data(self, data_file):
        """
        Load and process data directly from the CSV file format
        """
        print(f"Loading data from {data_file}")
        data = pd.read_csv(data_file)
        
        # No need to rename columns since they match expected format
        return data

    def train_model(self, data_file, data_percentage, data_folder=None):
        print(f"Training model for patient {self.patient} with {data_percentage}% of data...")
        
        # Load data
        data = self.load_data(data_file)
        
        # Process data (values are already in the correct format from CSV)
        angles = data['Angle'].values
        force_filtered = data['Force'].values
        cop_data = data['CoP'].values
        time_data = data['Time'].values  # Renamed from time to time_data
        forces_derivative = data['Force_Derivative'].values
        angles_derivative = data['Angle_Derivative'].values
        cop = cop_data

        # Get phase estimation
        gait_phases, gait_progress, start_time, end_time, stance_mask = self.offline_phase_estimator(time_data, force_filtered)

        # Create mask and ensure all arrays have the same length
        mask = (time_data >= start_time) & (time_data <= end_time)
        min_length = min(len(mask), len(gait_phases))
        
        # Truncate all arrays to the minimum length
        mask = mask[:min_length]
        gait_phases = gait_phases[:min_length]
        gait_progress = gait_progress[:min_length]
        time_data = time_data[:min_length]
        force_filtered = force_filtered[:min_length]
        angles = angles[:min_length]
        forces_derivative = forces_derivative[:min_length]
        angles_derivative = angles_derivative[:min_length]
        cop = cop[:min_length]

        # Apply mask to all data
        adjusted_time = time_data[mask]
        adjusted_force = force_filtered[mask]
        adjusted_angle = angles[mask]
        adjusted_force_derivatives = forces_derivative[mask]
        adjusted_angle_derivatives = angles_derivative[mask]
        adjusted_cop = cop[mask]
        
        # Get matching portions of phases and progress
        adjusted_gait_phases = gait_phases[mask]
        adjusted_gait_progress = gait_progress[mask]
        
        # Get filtered data using create_dataset_per_cycles
        (filtered_force, filtered_force_d, filtered_angle, filtered_angle_d, 
         filtered_cop, filtered_time, filtered_phase, filtered_progress) = self.create_dataset_per_cycles(
            adjusted_force, adjusted_force_derivatives, adjusted_angle,
            adjusted_angle_derivatives, adjusted_cop, adjusted_time, 
            adjusted_gait_phases, adjusted_gait_progress
        )

        # Save processed data
        print(f"Saving processed data to {self.labels_path}")
        processed_data = pd.DataFrame({
            'Time': filtered_time,
            'Force': filtered_force,
            'Force_Derivative': filtered_force_d,
            'Angle': filtered_angle,
            'Angle_Derivative': filtered_angle_d,
            'CoP': filtered_cop,
            'Gait_Progress': filtered_progress,
            'Phase': filtered_phase
        })
        processed_data.to_csv(self.labels_path, index=False)

        # Create and train model
        X = np.column_stack((filtered_force, filtered_force_d, filtered_angle, filtered_angle_d, filtered_cop))
        y = np.array(filtered_progress)

        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dump(scaler, self.scaler_path)
        self.logger.info(f"Scaler saved to {self.scaler_path}")

        X = X_scaled.reshape((X.shape[0], 1, X.shape[1]))

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Training
        self.logger.info("Training the LSTM model...")
        initial_time = times.time()

        history = model.fit(
            X_train, y_train,
            epochs=1000,
            batch_size=32,
            verbose=1,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        final_time = times.time()
        training_duration = final_time - initial_time
        self.logger.info(f"Training time: {training_duration:.2f} seconds")

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.logger.info(f"Performance Metrics:\nMSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('LSTM Training & Validation Loss')
        plt.legend()
        loss_plot_path = os.path.join(data_folder, "results", self.current_results_folder, "lstm_loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()
        # Save model
        self.logger.info(f"Saving LSTM model to {self.model_path}...")
        # Save model with percentage in filename (changed format)
        model_path_with_pct = os.path.join(self.current_results_folder, 
                                          f"{self.patient}_modelLSTM_{data_percentage}pct")
        try:
            model.save(model_path_with_pct)  # Removed save_format parameter
            print(f"Model saved to {model_path_with_pct}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
        # Save scaler with percentage in filename
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaler_path_with_pct = os.path.join(self.current_results_folder, 
                                           f"{self.patient}_scalerLSTM_{data_percentage}pct_{now}.pkl")
        dump(scaler, scaler_path_with_pct)
        self.modelLoaded = True

    # Retourner un dictionnaire avec les résultats
        return {
            'data_percentage': data_percentage,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'training_duration': training_duration,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'total_epochs': len(history.history['loss'])
        }

    def load_model(self):
        """Load pre-trained model and scaler"""
        try:
            self.model = load_model(self.model_path)
            self.scaler = load(self.scaler_path)
            self.modelLoaded = True
            self.logger.info(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.modelLoaded = False
            return False

    def estimate_phase(self):
        """Estimate gait phase from current sensor data"""
        if self.modelLoaded and self.angleUpdated and self.forceUpdated:
            current_input = [
                self.ground_force, self.ground_force_derivative,
                self.ankle_angle, self.ankle_angle_derivative,
                self.cop
            ]
            
            # Normalize input
            current_input_scaled = self.scaler.transform([current_input])
            current_input_scaled = current_input_scaled.reshape(1, 1, 5)
            
            # Make prediction
            new_phase = float(self.model.predict(current_input_scaled, verbose=0)[0])
            
            # Store results
            timestamp = times.time()
            self.results['timestamps'].append(timestamp)
            self.results['gait_percentage'].append(int(new_phase))
            self.results['force_derivative'].append(self.ground_force_derivative)
            self.results['angle_derivative'].append(self.ankle_angle_derivative)
            
            # Determine stance/swing phase
            phase_indicator = 100 if self.ground_force == 0 else 0
            self.results['stance_swing_phase'].append(phase_indicator)
            
            # Reset update flags
            self.angleUpdated = False
            self.forceUpdated = False
            
            return {
                'gait_percentage': int(new_phase),
                'force_derivative': self.ground_force_derivative,
                'angle_derivative': self.ankle_angle_derivative,
                'stance_swing_phase': phase_indicator,
                'timestamp': timestamp
            }
        
        return None

    def process_sensor_data(self, angle_deg, normalized_force, cop_x):
        """Process new sensor data and return gait phase estimation"""
        self.update_angle_data(angle_deg)
        self.update_force_data(normalized_force, cop_x)
        return self.estimate_phase()

    def get_results(self):
        """Get all stored results"""
        return self.results.copy()

    def save_results(self, data_folder, filename=None):
        """Save results to file"""
        if filename is None:
            filename = os.path.join(data_folder, "logs", f"{self.patient}_results.json")

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")

    def clear_results(self):
        """Clear stored results"""
        self.results = {
            'timestamps': [],
            'gait_percentage': [],
            'force_derivative': [],
            'angle_derivative': [],
            'stance_swing_phase': []
        }

    def train_with_multiple_percentages(self, data_file):
        percentages = [10, 20, 40, 60, 80, 100]
        all_results = []
        
        for pct in percentages:
            print(f"\nTraining with {pct}% of data...")
            results = self.train_model(data_file, pct, data_folder=self.data_folder)  # Passez data_folder ici
            all_results.append(results)
            
            print(f"\nResults for {pct}% of data:")
            print(f"MSE: {results['mse']:.4f}")
            print(f"RMSE: {results['rmse']:.4f}")
            print(f"MAE: {results['mae']:.4f}")
            print(f"R²: {results['r2']:.4f}")
            print(f"Training duration: {results['training_duration']:.2f} seconds")
            print(f"Final training loss: {results['final_train_loss']:.4f}")
            print(f"Final validation loss: {results['final_val_loss']:.4f}")
            print("----------------------------------------")

        # Save overall results
        results_file = os.path.join(self.current_results_folder, f"{self.patient}_overall_results.csv")
        overall_results = [{
            'data_percentage': r['data_percentage'],
            'mse': r['mse'],
            'rmse': r['rmse'],
            'mae': r['mae'],
            'r2': r['r2'],
            'training_duration': r['training_duration'],
            'final_train_loss': r['final_train_loss'],
            'final_val_loss': r['final_val_loss'],
            'total_epochs': r['total_epochs']
        } for r in all_results]
        pd.DataFrame(overall_results).to_csv(results_file, index=False)

        # Create and save comparative plots
        plt.figure(figsize=(15, 10))
        metrics = ['mse', 'rmse', 'mae', 'r2']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            values = [r[metric] for r in all_results]
            plt.plot(percentages, values, 'o-')
            plt.xlabel('Data Percentage')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} vs Data Percentage')
            plt.grid(True)
        
        plt.tight_layout()
        metrics_plot_path = os.path.join(self.current_results_folder, f"{self.patient}_metrics_comparison.png")
        plt.savefig(metrics_plot_path)
        plt.close()

        return all_results

def main():
    # Dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Chemin vers le dossier de données
    data_folder = os.path.join(project_root, "train_data_filtered_labeled_csv")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Fichier de données
    data_file = os.path.join(data_folder, "subject7_labelsLSTM.csv")

    # Initialiser et entraîner le modèle
    estimator = GaitPhaseEstimator(data_folder, patient_id="subject7")

    if os.path.exists(data_file):
        results = estimator.train_with_multiple_percentages(data_file)
        print("Entraînement terminé. Les résultats ont été sauvegardés dans:", estimator.current_results_folder)
    else:
        print(f"Erreur: Le fichier {data_file} n'existe pas")

if __name__ == '__main__':
    main()