import os
import sys

# Importer les classes des modèles
from RNN_model_WOSC import GaitPhaseEstimator as RNNModel
from CNN_model_WOSC import GaitPhaseEstimator as CNNModel
from NRAX_model_WOSC import GaitPhaseEstimator as NRAXModel
from LSTM_model_WOSC import GaitPhaseEstimator as LSTMModel

def main(subject_number):
    # Dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Chemin vers le dossier de données
    data_folder = os.path.join(project_root, "train_data_filtered_labeled_csv")
    
    # Fichier de données
    data_file = os.path.join(data_folder, f"subject{subject_number}_labelsRNN.csv")
    
    # Exécuter le modèle RNN
    print(f"Training RNN model for subject {subject_number}...")
    rnn_estimator = RNNModel(data_folder, patient_id=f"subject{subject_number}")
    if os.path.exists(data_file):
        rnn_estimator.train_model(data_file, data_percentage=100)  # Vous pouvez ajuster le pourcentage ici
    else:
        print(f"Data file for subject {subject_number} not found.")

    # Exécuter le modèle CNN
    print(f"Training CNN model for subject {subject_number}...")
    cnn_estimator = CNNModel(data_folder, patient_id=f"subject{subject_number}")
    data_file = os.path.join(data_folder, f"subject{subject_number}_labelsCNN.csv")
    if os.path.exists(data_file):
        cnn_estimator.train_model(data_file, data_percentage=100)  # Ajustez le pourcentage ici
    else:
        print(f"Data file for subject {subject_number} not found.")

    # Exécuter le modèle NRAX
    print(f"Training NRAX model for subject {subject_number}...")
    nr_ax_estimator = NRAXModel(data_folder, patient_id=f"subject{subject_number}")
    data_file = os.path.join(data_folder, f"subject{subject_number}_labelsNRAX.csv")
    if os.path.exists(data_file):
        nr_ax_estimator.train_model(data_file, data_percentage=100)  # Ajustez le pourcentage ici
    else:
        print(f"Data file for subject {subject_number} not found.")

    # Exécuter le modèle LSTM
    print(f"Training LSTM model for subject {subject_number}...")
    lstm_estimator = LSTMModel(data_folder, patient_id=f"subject{subject_number}")
    data_file = os.path.join(data_folder, f"subject{subject_number}_labelsLSTM.csv")
    if os.path.exists(data_file):
        lstm_estimator.train_model(data_file, data_percentage=100)  # Ajustez le pourcentage ici
    else:
        print(f"Data file for subject {subject_number} not found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_models.py <subject_number>")
        sys.exit(1)

    subject_number = sys.argv[1]
    main(subject_number)