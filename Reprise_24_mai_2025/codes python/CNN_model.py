#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump, load
from collections import deque
import time
import logging
from datetime import datetime

class GaitPhaseEstimator:
    def __init__(self, data_folder, patient_id="subject1", samples_size=10):
        # Setup paths
        self.patient = patient_id
        self.base_path = os.path.abspath(data_folder)  # Convert to absolute path
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)  # Added exist_ok=True
            
        self.labels_path = os.path.join(self.base_path, f"{self.patient}_labelsCNN.csv")
        self.model_path = os.path.join(self.base_path, f"{self.patient}_modelCNN.keras")
        self.scaler_path = os.path.join(self.base_path, f"{self.patient}_modelCNN_scaler.pkl")
        self.log_file_path = os.path.join(self.base_path, f"{self.patient}_modelCNN_log.txt")

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
        self.current_results_folder = os.path.join(self.results_folder, f"training_{now}")
        os.makedirs(self.current_results_folder)
            

    def setup_logger(self):
        logging.basicConfig(
            filename=self.log_file_path,
            level=logging.INFO,
            format='[%(asctime)s] %(message)s'
        )

    def load_data(self, data_file):
        """
        Load and process data directly from the CSV file format
        """
        print(f"Loading data from {data_file}")
        data = pd.read_csv(data_file)
        
        # No need to rename columns since they match expected format
        return data

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
        print("Percentage of data used to train: " + str(ptg_data))

        # Returns filtered signals
        return filtered_force, filtered_force_d, filtered_angle, filtered_angle_d, filtered_cop, filtered_time, filtered_phase, filtered_progress 
        
    def train_model(self, data_file, data_percentage=100):
        print(f"Training model for patient {self.patient} with {data_percentage}% of data...")
        
        # Load data
        data = self.load_data(data_file)
        
        # Process data (values are already in the correct format from CSV)
        angle_data = data['Angle'].values
        interpolated_forces = data['Force'].values
        cop_data = data['CoP'].values
        time_data = data['Time'].values  # Renamed from time to time_data

        ''' Ground Force Filter '''
        # Force filter and correction
        force_filtered = filtfilt(self.b, self.a, interpolated_forces)
        force_filtered = np.array([f if f >= self.force_threshold else 0.0 for f in force_filtered])

        # Derivative of force
        f_derivative = np.diff(force_filtered)
        f_derivative = np.append(0, f_derivative)

        forces = force_filtered
        forces_derivative = f_derivative

        ''' Angle Filter '''
        angle_filtered = filtfilt(self.b, self.a, angle_data)
        
        vel_rps = np.diff(angle_filtered)
        vel_rps = np.append(0, vel_rps)
        
        angles = angle_filtered
        angles_derivative = vel_rps

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
        forces = forces[:min_length]
        angles = angles[:min_length]
        forces_derivative = forces_derivative[:min_length]
        angles_derivative = angles_derivative[:min_length]
        cop = cop[:min_length]

        # Apply mask to all data
        adjusted_time = time_data[mask]
        adjusted_force = forces[mask]
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

        # Data Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dump(scaler, self.scaler_path)
        print(f"Scaler saved to {self.scaler_path}")

        # Create Sequences
        def create_sequences(data, labels, seq_length):
            sequences, label_sequences = [], []
            for i in range(len(data) - seq_length + 1):
                sequences.append(data[i:i + seq_length])
                label_sequences.append(labels[i + seq_length - 1])
            return np.array(sequences), np.array(label_sequences)

        X_seq, y_seq = create_sequences(X_scaled, y, self.sequence_length)

        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        # Adjust training data size based on percentage
        train_size = int(len(X_train) * (data_percentage/100))
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]

        # CNN + LSTM Model
        model = Sequential([
            Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            MaxPooling1D(pool_size=2), Dropout(0.3), 
            
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2), Dropout(0.3),
            
            LSTM(50, return_sequences=False), Dropout(0.3),
            
            Dense(32, activation='relu'), Dropout(0.2), 

            Dense(1, activation='linear')
        ])

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Training
        print("Training the CNN+LSTM model...")

        initial_time = time.time()

        history = model.fit(
            X_train, y_train,
            epochs=1000,
            batch_size=32,
            verbose=1,  # Changed from 0 to 1
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        final_time = time.time()

        # Calculate and print elapsed time
        training_duration = final_time - initial_time
        print(f"Training time: {training_duration:.2f} seconds")

        # Loss History
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Model Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Performance Metrics:\nMSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        print("Last Train and Validation Loss for CNN+LSTM Model:")
        print(f"Train Loss={train_loss[-1]:.4f}, Validation Loss={val_loss[-1]:.4f}")

        # Save model with percentage in filename (changed format)
        model_path_with_pct = os.path.join(self.current_results_folder, 
                                          f"{self.patient}_modelCNN_{data_percentage}pct")
        try:
            model.save(model_path_with_pct)  # Removed save_format parameter
            print(f"Model saved to {model_path_with_pct}")
        except Exception as e:
            print(f"Error saving model: {e}")
            
        # Save scaler with percentage in filename
        scaler_path_with_pct = os.path.join(self.current_results_folder, 
                                           f"{self.patient}_scalerCNN_{data_percentage}pct.pkl")
        dump(scaler, scaler_path_with_pct)

        # Save training history
        history_dict = history.history
        history_dict['epochs'] = list(range(1, len(train_loss) + 1))
        history_df = pd.DataFrame(history_dict)
        history_path = os.path.join(self.current_results_folder, 
                                   f"{self.patient}_history_{data_percentage}pct.csv")
        history_df.to_csv(history_path, index=False)

        return {
            'data_percentage': data_percentage,
            'training_duration': training_duration,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'final_train_loss': train_loss[-1],
            'final_val_loss': val_loss[-1],
            'total_epochs': len(train_loss),
            'train_loss': train_loss,
            'val_loss': val_loss
        }

    def predict(self, input_data):
        """
        Make predictions on new data
        input_data should be a numpy array of shape (sequence_length, 5)
        containing [force, force_derivative, angle, angle_derivative, cop]
        """
        if not self.modelLoaded:
            try:
                self.model = load_model(os.path.dirname(self.model_path))  # Updated path handling
                self.scaler = load(self.scaler_path)
                self.modelLoaded = True
            except FileNotFoundError:
                raise Exception("Model or scaler not found. Please train the model first.")

        # Scale input data
        scaled_data = self.scaler.transform(input_data)
        scaled_data = scaled_data.reshape(1, self.sequence_length, 5)

        # Make prediction
        prediction = float(self.model.predict(scaled_data, verbose=0)[-1])
        return prediction

    def train_with_multiple_percentages(self, data_file):
        percentages = [10, 20, 40, 60, 80, 100]
        all_results = []
        
        for pct in percentages:
            print(f"\nTraining with {pct}% of data...")
            results = self.train_model(data_file, pct)
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
        results_file = os.path.join(self.current_results_folder, f"{self.patient_id}_overall_results.csv")
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
        metrics_plot_path = os.path.join(self.current_results_folder, f"{self.patient_id}_metrics_comparison.png")
        plt.savefig(metrics_plot_path)
        plt.close()

        return all_results

def main():
    # Dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Chemin vers le dossier de données
    data_folder = os.path.join(project_root, "train_data_labeled_csv")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Fichier de données
    data_file = os.path.join(data_folder, "subject1_labelsCNN.csv")
    
    # Initialiser et entraîner le modèle
    estimator = GaitPhaseEstimator(data_folder, patient_id="subject1")
    
    if os.path.exists(data_file):
        results = estimator.train_with_multiple_percentages(data_file)
        print("Entraînement terminé. Les résultats ont été sauvegardés dans:", estimator.current_results_folder)
    else:
        print(f"Erreur: Le fichier {data_file} n'existe pas")

if __name__ == "__main__":
    main()

