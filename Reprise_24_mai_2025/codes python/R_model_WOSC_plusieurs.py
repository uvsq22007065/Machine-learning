#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import time
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import butter, filtfilt
import logging

class GaitPhaseEstimator:
    def __init__(self, data_folder, patient_id="subject9", samples_size=10):
        self.patient = patient_id
        self.base_path = os.path.abspath(data_folder)
        
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)

        # Create results folder structure
        self.results_folder = os.path.join(os.path.dirname(data_folder), "results")
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_results_folder = os.path.join(self.results_folder, f"Classical_{patient_id}_training_{now}")
        if not os.path.exists(self.current_results_folder):
            os.makedirs(self.current_results_folder)

        # Initialize model variables
        self.modelLoaded = False
        self.model = None
        self.model_GBR = None
        self.model_RFR = None
        self.model_VR = None
        self.force_threshold = 0.04
        self.samples_size = samples_size

        # Add filter parameters
        self.fs = 100
        fc = 5
        self.b, self.a = butter(3, fc / (self.fs / 2), btype='low')
        fc_vel = 10
        self.b_vel, self.a_vel = butter(3, fc_vel / (self.fs / 2), btype='low')

        # Initialize log file path
        self.log_file_path = os.path.join(data_folder, f"{patient_id}_gait_estimation.log")
        self.setup_logger()
        
        self.labels_path = os.path.join(self.current_results_folder, f"{self.patient}_labelsR.csv")

    def setup_logger(self):
        """Setup logging configuration"""
        logging.basicConfig(
            filename=self.log_file_path,
            level=logging.INFO,
            format='[%(asctime)s] %(message)s'
        )

    def load_data(self, data_file):
        """Load and process data from CSV file"""
        print(f"Loading data from {data_file}")
        data = pd.read_csv(data_file)
        return data

    def train_model(self, data_file, data_percentage):
        print(f"Training model for patient {self.patient} with {data_percentage}% of data...")
        
        # Load data
        data = self.load_data(data_file)
        
        # Process data
        angles = data['Angle'].values
        force_filtered = data['Force'].values
        cop_data = data['CoP'].values
        time_data = data['Time'].values
        forces_derivative = data['Force_Derivative'].values
        angles_derivative = data['Angle_Derivative'].values
        
        # Suppression de l'appel à offline_phase_estimator
        # gait_phases, gait_progress, start_time, end_time, stance_mask = self.offline_phase_estimator(time_data, force_filtered)

        # On suppose que les labels sont déjà dans le fichier
        gait_phases = data['Phase'].values
        gait_progress = data['Gait_Progress'].values

        # Create mask and ensure all arrays have the same length
        # On suppose que tout le vecteur est utilisable
        min_length = min(len(time_data), len(gait_phases))
        mask = np.ones(min_length, dtype=bool)
        
        # Truncate and apply mask to all data
        filtered_data = self.process_and_filter_data(
            mask, min_length, time_data, force_filtered, angles,
            forces_derivative, angles_derivative, cop_data,
            gait_phases, gait_progress
        )

        # Save processed data
        processed_data = pd.DataFrame({
            'Time': filtered_data['time'],
            'Force': filtered_data['force'],
            'Force_Derivative': filtered_data['force_d'],
            'Angle': filtered_data['angle'],
            'Angle_Derivative': filtered_data['angle_d'],
            'CoP': filtered_data['cop'],
            'Gait_Progress': filtered_data['progress'],
            'Phase': filtered_data['phase']
        })
        processed_data.to_csv(os.path.join(self.current_results_folder, 
                            f"{self.patient}_processed_data_{data_percentage}pct.csv"), index=False)

        # Prepare training data
        X = np.column_stack((filtered_data['force'], filtered_data['force_d'], 
                           filtered_data['angle'], filtered_data['angle_d'], 
                           filtered_data['cop']))
        y = np.array(filtered_data['progress'])

        # Split and adjust data size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_size = int(len(X_train) * (data_percentage/100))
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]

        # Train and evaluate models with detailed logging and visualization
        results = {}
        for name, model in self.get_models().items():
            print(f"\nTraining {name} model...")
            logging.info(f"Starting training for {name} model with {data_percentage}% data")
            
            train_loss, val_loss, epochs = [], [], []

            initial_time = time.time()

            model.fit(X_train, y_train)
            training_duration = time.time() - initial_time

            # Si le modèle supporte staged_predict (ex: GradientBoostingRegressor)
            if hasattr(model, "staged_predict"):
                for i, (y_train_pred, y_val_pred) in enumerate(zip(model.staged_predict(X_train), model.staged_predict(X_test)), 1):
                    train_loss.append(mean_squared_error(y_train, y_train_pred))
                    val_loss.append(mean_squared_error(y_test, y_val_pred))
                    epochs.append(i)
            else:
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_test)
                train_loss = [mean_squared_error(y_train, y_train_pred)]
                val_loss = [mean_squared_error(y_test, y_val_pred)]
                epochs = [1]
                y_pred = y_val_pred  # Pour la suite (évaluation et sauvegarde)

            # Pour GradientBoosting, prendre la dernière prédiction
            if hasattr(model, "staged_predict"):
                y_pred = list(model.staged_predict(X_test))[-1]

            metrics = self.calculate_metrics(y_test, y_pred)
            metrics['training_duration'] = training_duration
            metrics['final_train_loss'] = train_loss[-1]
            metrics['final_val_loss'] = val_loss[-1]
            results[name] = metrics

            self.save_model_artifacts(name, model, data_percentage, y_test, y_pred)
            self.plot_predictions(y_test, y_pred, name, data_percentage)

            # 🔁 Ajout : tracer train/val loss
            # self.plot_training_history(train_loss, val_loss, epochs, name, data_percentage)
            # 🔽 Sauvegarde complète de la courbe de loss dans un CSV
            loss_df = pd.DataFrame({
                'epoch': epochs,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            loss_csv_path = os.path.join(
                self.current_results_folder,
                f"{self.patient}_{name.replace(' ', '_')}_{data_percentage}pct_loss_curve.csv"
                )
            loss_df.to_csv(loss_csv_path, index=False)

            logging.info(f"Model: {name} - Data: {data_percentage}%")
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}")

        # Save detailed results and create additional visualizations
        self.save_detailed_results(results, data_percentage)
        # self.plot_performance_metrics(results, data_percentage)

        return results

    def process_and_filter_data(self, mask, min_length, time_data, force_filtered, angles,
                              forces_derivative, angles_derivative, cop_data, 
                              gait_phases, gait_progress):
        # Truncate arrays
        arrays = {
            'time': time_data[:min_length][mask],
            'force': force_filtered[:min_length][mask],
            'angle': angles[:min_length][mask],
            'force_d': forces_derivative[:min_length][mask],
            'angle_d': angles_derivative[:min_length][mask],
            'cop': cop_data[:min_length][mask]
        }
        
        # Get filtered data using create_dataset_per_cycles
        filtered_data = self.create_dataset_per_cycles(
            arrays['force'], arrays['force_d'], arrays['angle'],
            arrays['angle_d'], arrays['cop'], arrays['time'],
            gait_phases[mask], gait_progress[mask]
        )
        
        return dict(zip(['force', 'force_d', 'angle', 'angle_d', 'cop', 'time', 'phase', 'progress'], 
                       filtered_data))

    def get_models(self):
        return {
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=9, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Voting Regressor': VotingRegressor(estimators=[
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=9, random_state=42))
            ])
        }

    def calculate_metrics(self, y_true, y_pred):
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def save_model_artifacts(self, model_name, model, data_percentage, y_test, y_pred):
        # Save model
        model_path = os.path.join(self.current_results_folder, 
                                f"{self.patient}_{model_name.replace(' ', '_')}_{data_percentage}pct.joblib")
        joblib.dump(model, model_path)

        # Save predictions
        predictions_df = pd.DataFrame({
            'True_Progress': y_test,
            'Predicted_Progress': y_pred
        })
        pred_path = os.path.join(self.current_results_folder, 
                               f"{self.patient}_{model_name.replace(' ', '_')}_{data_percentage}pct_predictions.csv")
        predictions_df.to_csv(pred_path, index=False)

    def train_with_multiple_percentages(self, data_file):
        percentages = [10, 20, 40, 60, 80, 100]
        all_results = []
        
        for pct in percentages:
            print(f"\nTraining with {pct}% of data...")
            results = self.train_model(data_file, pct)
            
            # Compile results for all models
            for model_name, model_results in results.items():
                result_entry = {
                    'data_percentage': pct,
                    'model': model_name,
                    'mse': model_results['mse'],
                    'rmse': model_results['rmse'],
                    'mae': model_results['mae'],
                    'r2': model_results['r2'],
                    'training_duration': model_results['training_duration'],
                    'final_train_loss': model_results.get('final_train_loss', None),
                    'final_val_loss': model_results.get('final_val_loss', None)
                }
                all_results.append(result_entry)
                
                print(f"\nResults for {model_name} with {pct}% of data:")
                print(f"MSE: {model_results['mse']:.4f}")
                print(f"RMSE: {model_results['rmse']:.4f}")
                print(f"MAE: {model_results['mae']:.4f}")
                print(f"R²: {model_results['r2']:.4f}")
                print(f"Training duration: {model_results['training_duration']:.2f} seconds")
                print("----------------------------------------")

        # Save overall results
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(self.current_results_folder, f"{self.patient}_overall_results.csv")
        results_df.to_csv(results_file, index=False)

        # Create comparative plots
        self.plot_metrics_comparison(results_df)
        
        return all_results

    def plot_metrics_comparison(self, results_df):
        metrics = ['mse', 'rmse', 'mae', 'r2']
        models = results_df['model'].unique()
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            for model in models:
                model_data = results_df[results_df['model'] == model]
                plt.plot(model_data['data_percentage'], model_data[metric], 'o-', label=model)
            plt.xlabel('Data Percentage')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} vs Data Percentage')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        metrics_plot_path = os.path.join(self.current_results_folder, f"{self.patient}_metrics_comparison.svg")
        plt.savefig(metrics_plot_path)
        plt.close()

    def offline_phase_estimator(self, time, interpolated_forces_raw):
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

    def create_dataset_per_cycles(self, adjusted_force, adjusted_force_derivatives, adjusted_angle,
                                adjusted_angle_derivatives, adjusted_cop, adjusted_time, gait_phases, gait_progress):
        cycles_force = []
        cycles_force_deriv = []
        cycles_angle = []
        cycles_angle_deriv = []
        cycles_cop = []
        cycles_time = []
        cycles_phase = []
        cycles_progress = []

        current_cycle = {'force': [], 'force_d': [], 'angle': [], 'angle_d': [],
                        'cop': [], 'time': [], 'phase': [], 'progress': []}

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

                current_cycle = {'force': [], 'force_d': [], 'angle': [], 'angle_d': [], 
                               'cop': [], 'time': [], 'phase': [], 'progress': []}

        if len(cycles_angle) == 0:
            return adjusted_force, adjusted_force_derivatives, adjusted_angle, adjusted_angle_derivatives, adjusted_cop, adjusted_time, gait_phases, gait_progress

        # Suppression de l'appel à select_consistent_cycles
        # selected_indices_angle = self.select_consistent_cycles(cycles_angle, cycles_phase, percentage=90)
        # selected_indices_force = self.select_consistent_cycles(cycles_force, cycles_phase, percentage=90)
        # selected_indices = list(set(selected_indices_angle) & set(selected_indices_force))

        # On considère que tous les cycles sont consistents
        selected_indices = list(range(len(cycles_angle)))

        filtered_force = np.concatenate([cycles_force[i] for i in selected_indices])
        filtered_force_d = np.concatenate([cycles_force_deriv[i] for i in selected_indices])
        filtered_angle = np.concatenate([cycles_angle[i] for i in selected_indices])
        filtered_angle_d = np.concatenate([cycles_angle_deriv[i] for i in selected_indices])
        filtered_cop = np.concatenate([cycles_cop[i] for i in selected_indices])
        filtered_time = np.concatenate([cycles_time[i] for i in selected_indices])
        filtered_phase = np.concatenate([['stance_phase' if p < 60 else 'swing_phase' for p in cycles_phase[i]] 
                                       for i in selected_indices])
        filtered_progress = np.concatenate([cycles_progress[i] for i in selected_indices])

        print("Percentage of data used to train: " + str((len(filtered_time)*100.0/len(adjusted_time))))

        return filtered_force, filtered_force_d, filtered_angle, filtered_angle_d, filtered_cop, filtered_time, filtered_phase, filtered_progress

    def plot_predictions(self, y_true, y_pred, model_name, data_percentage):
        """Create detailed prediction plots"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')  # Perfect prediction line
        plt.xlabel('True Progress')
        plt.ylabel('Predicted Progress')
        plt.title(f'{model_name} Predictions vs True Values ({data_percentage}% data)')
        
        # Add metrics text to plot
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plot_path = os.path.join(self.current_results_folder, 
                                f"{self.patient}_{model_name.replace(' ', '_')}_{data_percentage}pct_predictions.svg")
        plt.savefig(plot_path)
        plt.close()

    def predict(self, input_data, model_name='Gradient Boosting'):
        """Make predictions using the specified model"""
        model_path = os.path.join(self.current_results_folder, 
                                f"{self.patient}_{model_name.replace(' ', '_')}_100pct.joblib")
        try:
            model = joblib.load(model_path)
            prediction = float(model.predict([input_data])[0])
            logging.info(f"Prediction made using {model_name}: {prediction}")
            return prediction
        except FileNotFoundError:
            logging.error(f"Model {model_name} not found at {model_path}")
            raise Exception(f"Model {model_name} not found. Please train the model first.")

    def plot_training_history(self, train_metrics, val_metrics, epochs, model_name, data_percentage):
        """Plot training history for a model"""
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_metrics, label='Training Loss')
        plt.plot(epochs, val_metrics, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training History - {model_name} ({data_percentage}% data)')
        plt.legend()
        plt.grid(True)
        
        history_plot_path = os.path.join(self.current_results_folder, 
                                       f"{self.patient}_{model_name.replace(' ', '_')}_{data_percentage}pct_history.svg")
        plt.savefig(history_plot_path)
        plt.close()

    def plot_performance_metrics(self, results, data_percentage):
        """Plot detailed performance metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        for model_name in results.keys():
            metrics = results[model_name]
            ax1.plot(['MSE'], [metrics['mse']], 'o-', label=model_name)
            ax2.plot(['RMSE'], [metrics['rmse']], 'o-', label=model_name)
            ax3.plot(['MAE'], [metrics['mae']], 'o-', label=model_name)
            ax4.plot(['R²'], [metrics['r2']], 'o-', label=model_name)
        
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Mean Squared Error', 'Root Mean Squared Error', 
                 'Mean Absolute Error', 'R² Score']
        
        for ax, title in zip(axes, titles):
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        metrics_detail_path = os.path.join(self.current_results_folder, 
                                         f"{self.patient}_detailed_metrics_{data_percentage}pct.svg")
        plt.savefig(metrics_detail_path)
        plt.close()

    def save_detailed_results(self, results, data_percentage):
        """Save detailed results to CSV"""
        detailed_results = []
        
        for model_name, metrics in results.items():
            result_entry = {
                'model': model_name,
                'data_percentage': data_percentage,
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'training_duration': metrics['training_duration']
            }
            detailed_results.append(result_entry)
        
        results_df = pd.DataFrame(detailed_results)
        detailed_results_path = os.path.join(self.current_results_folder, 
                                           f"{self.patient}_detailed_results_{data_percentage}pct.csv")
        results_df.to_csv(detailed_results_path, index=False)

def main():
    # Dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "train_data_filtered_labeled_csv")

    # Liste des sujets à traiter
    subjects = ["subject1", "subject2", "subject3", "subject4", "subject5","subject6", "subject7", "subject8","subject9", "subject10", "subject11"]

    for subject in subjects:
        data_file = os.path.join(data_folder, f"{subject}_labelsR.csv")
        estimator = GaitPhaseEstimator(data_folder, patient_id=subject)

        if os.path.exists(data_file):
            print(f"\n=== Lancement pour {subject} ===")
            results = estimator.train_with_multiple_percentages(data_file)
            print("Entraînement terminé. Les résultats ont été sauvegardés dans:", estimator.current_results_folder)
        else:
            print(f"Erreur: Le fichier {data_file} n'existe pas")

if __name__ == "__main__":
    main()
