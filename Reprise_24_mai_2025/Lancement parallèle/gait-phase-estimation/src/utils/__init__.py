import os
from LSTM_model_WOSC import GaitPhaseEstimator as LSTMEstimator
from CNN_model_WOSC import GaitPhaseEstimator as CNNEstimator
from RNN_model_WOSC import GaitPhaseEstimator as RNNEstimator
from NRAX_model_WOSC import GaitPhaseEstimator as NRAXEstimator

def run_models(subject_id, data_folder, data_percentage=100):
    # Initialize model estimators
    lstm_estimator = LSTMEstimator(data_folder, patient_id=subject_id)
    cnn_estimator = CNNEstimator(data_folder, patient_id=subject_id)
    rnn_estimator = RNNEstimator(data_folder, patient_id=subject_id)
    nr_ax_estimator = NRAXEstimator(data_folder, patient_id=subject_id)

    # Define the data file path
    data_file = os.path.join(data_folder, f"{subject_id}_labels{subject_id}.csv")

    # Check if the data file exists
    if os.path.exists(data_file):
        print(f"Running models for subject: {subject_id}")

        # Train each model
        lstm_estimator.train_model(data_file, data_percentage)
        cnn_estimator.train_model(data_file, data_percentage)
        rnn_estimator.train_model(data_file, data_percentage)
        nr_ax_estimator.train_model(data_file, data_percentage)

        print("All models have been trained successfully.")
    else:
        print(f"Data file for subject {subject_id} does not exist: {data_file}")

def main():
    # Specify the subject ID and data folder
    subject_id = input("Enter the subject ID (e.g., subject2): ")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_root, "train_data_filtered_labeled_csv")

    run_models(subject_id, data_folder)

if __name__ == "__main__":
    main()