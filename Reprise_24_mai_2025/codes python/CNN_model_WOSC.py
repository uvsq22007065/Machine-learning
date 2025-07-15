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
    def __init__(self, data_folder, patient_id="subject11", samples_size=10):
        # Setup paths
        self.patient = patient_id
        self.base_path = os.path.abspath(data_folder)  # Convert to absolute path
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)  # Added exist_ok=True

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
        self.current_results_folder = os.path.join(self.results_folder, f"CNN_{patient_id}_final_training_{now}_with_prediction")
        if not os.path.exists(self.current_results_folder):
            os.makedirs(self.current_results_folder)

        self.labels_path = os.path.join(self.current_results_folder, f"{self.patient}_labelsCNN.csv")
        self.model_path = os.path.join(self.current_results_folder, f"{self.patient}_modelCNN.keras")
        self.scaler_path = os.path.join(self.current_results_folder, f"{self.patient}_modelCNN_scaler.pkl")

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

        # Suppression de la sélection de cycles cohérents : on utilise tous les cycles extraits
        selected_indices = list(range(len(cycles_angle)))

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
        
    def train_model(self, data_file, data_percentage):
        print(f"Training model for patient {self.patient} with {data_percentage}% of data...")

        # Load data
        data = self.load_data(data_file)

        # Direct extraction (plus de mask ni de découpage)
        angles = data['Angle'].values
        force_filtered = data['Force'].values
        cop = data['CoP'].values
        time_data = data['Time'].values
        forces_derivative = data['Force_Derivative'].values
        angles_derivative = data['Angle_Derivative'].values
        gait_progress = data['Gait_Progress'].values
        gait_phases = data['Phase'].values

        # Plus de create_dataset_per_cycles, on suppose les cycles déjà propres
        filtered_force = force_filtered
        filtered_force_d = forces_derivative
        filtered_angle = angles
        filtered_angle_d = angles_derivative
        filtered_cop = cop
        filtered_time = time_data
        filtered_phase = gait_phases
        filtered_progress = gait_progress

        # Save processed data (optionnel ici, car déjà propre)
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
            verbose=1,
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
        predictions_df = pd.DataFrame({
            'True_Progress': y_test.flatten(),
            'Predicted_Progress': y_pred.flatten()
        })
        predictions_path = os.path.join(self.current_results_folder, f"{self.patient}_predictions_{data_percentage}pct.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Prédictions sauvegardées dans {predictions_path}")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Performance Metrics:\nMSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        print("Last Train and Validation Loss for CNN+LSTM Model:")
        print(f"Train Loss={train_loss[-1]:.4f}, Validation Loss={val_loss[-1]:.4f}")

        # Save model with percentage in filename
        model_path_with_pct = os.path.join(self.current_results_folder, f"{self.patient}_modelCNN_{data_percentage}pct")
        try:
            model.save(model_path_with_pct)
            print(f"Model saved to {model_path_with_pct}")
        except Exception as e:
            print(f"Error saving model: {e}")

        # Save scaler with percentage in filename
        scaler_path_with_pct = os.path.join(self.current_results_folder, f"{self.patient}_scalerCNN_{data_percentage}pct.pkl")
        dump(scaler, scaler_path_with_pct)

        # Save training history
        history_dict = history.history
        history_dict['epochs'] = list(range(1, len(train_loss) + 1))
        history_df = pd.DataFrame(history_dict)
        history_path = os.path.join(self.current_results_folder, f"{self.patient}_history_{data_percentage}pct.csv")
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
        metrics_plot_path = os.path.join(self.current_results_folder, f"{self.patient}_metrics_comparison.svg")
        plt.savefig(metrics_plot_path)
        plt.close()

def main():
    # Dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Chemin vers le dossier de données
    data_folder = os.path.join(project_root, "train_data_filtered_labeled_csv")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    # Fichier de données
    data_file = os.path.join(data_folder, "subject11_labelsCNN.csv")
    
    # Initialiser et entraîner le modèle
    estimator = GaitPhaseEstimator(data_folder, patient_id="subject11")

    if os.path.exists(data_file):
        results = estimator.train_with_multiple_percentages(data_file)
        print("Entraînement terminé. Les résultats ont été sauvegardés dans:", estimator.current_results_folder)
    else:
        print(f"Erreur: Le fichier {data_file} n'existe pas")

if __name__ == "__main__":
    main()

