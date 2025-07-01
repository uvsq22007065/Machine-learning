import os
import sys
from LSTM_model_WOSC import GaitPhaseEstimator as LSTMEstimator
from CNN_model_WOSC import GaitPhaseEstimator as CNNEstimator
from RNN_model_WOSC import GaitPhaseEstimator as RNNEstimator
from NRAX_model_WOSC import GaitPhaseEstimator as NRAXEstimator

def main(subject_number):
    # Dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Chemin vers le dossier de données
    data_folder = os.path.join(project_root, "train_data_filtered_labeled_csv")
    
    # Fichier de données
    data_file = os.path.join(data_folder, f"subject{subject_number}_labels.csv")

    # Vérifier si le fichier de données existe
    if not os.path.exists(data_file):
        print(f"Le fichier de données pour le sujet {subject_number} n'existe pas.")
        return

    # Initialiser et entraîner les modèles
    print(f"Démarrage de l'entraînement pour le sujet {subject_number}...")

    # LSTM Model
    lstm_estimator = LSTMEstimator(data_folder, patient_id=f"subject{subject_number}")
    lstm_estimator.train_with_multiple_percentages(data_file)

    # CNN Model
    cnn_estimator = CNNEstimator(data_folder, patient_id=f"subject{subject_number}")
    cnn_estimator.train_with_multiple_percentages(data_file)

    # RNN Model
    rnn_estimator = RNNEstimator(data_folder, patient_id=f"subject{subject_number}")
    rnn_estimator.train_with_multiple_percentages(data_file)

    # NRAX Model
    nr_ax_estimator = NRAXEstimator(data_folder, patient_id=f"subject{subject_number}")
    nr_ax_estimator.train_with_multiple_percentages(data_file)

    print(f"Entraînement terminé pour le sujet {subject_number}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_models.py <subject_number>")
        sys.exit(1)

    subject_number = sys.argv[1]
    main(subject_number)