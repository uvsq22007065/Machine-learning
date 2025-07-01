import os
from LSTM_model_WOSC import GaitPhaseEstimator as LSTMEstimator
from CNN_model_WOSC import GaitPhaseEstimator as CNNEstimator
from RNN_model_WOSC import GaitPhaseEstimator as RNNEstimator
from NRAX_model_WOSC import GaitPhaseEstimator as NRAXEstimator

def run_models(subject_id):
    # Dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Chemin vers le dossier de données
    data_folder = os.path.join(project_root, "train_data_filtered_labeled_csv")
    
    # Fichier de données
    data_file = os.path.join(data_folder, f"{subject_id}_labels{subject_id}.csv")  # Assurez-vous que le nom du fichier est correct

    # Initialiser et entraîner les modèles
    print(f"Starting training for subject: {subject_id}")

    # LSTM Model
    lstm_estimator = LSTMEstimator(data_folder, patient_id=subject_id)
    lstm_estimator.train_model(data_file, data_percentage=100)  # Vous pouvez ajuster le pourcentage de données ici

    # CNN Model
    cnn_estimator = CNNEstimator(data_folder, patient_id=subject_id)
    cnn_estimator.train_model(data_file, data_percentage=100)  # Vous pouvez ajuster le pourcentage de données ici

    # RNN Model
    rnn_estimator = RNNEstimator(data_folder, patient_id=subject_id)
    rnn_estimator.train_model(data_file, data_percentage=100)  # Vous pouvez ajuster le pourcentage de données ici

    # NRAX Model
    nrax_estimator = NRAXEstimator(data_folder, patient_id=subject_id)
    nrax_estimator.train_model(data_file, data_percentage=100)  # Vous pouvez ajuster le pourcentage de données ici

    print(f"Finished training for subject: {subject_id}")

if __name__ == "__main__":
    subject_id = input("Enter the subject ID (e.g., subject2): ")
    run_models(subject_id)