# Dossier racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Chemin vers le dossier de données
    data_folder = os.path.join(project_root, "train_data_filtered_labeled_csv")
    
    # Fichier de données
    data_file = os.path.join(data_folder, f"subject{subject_number}_labels.csv")  # Assurez-vous que le nom du fichier correspond

    # Initialiser et entraîner les modèles
    print(f"Starting training for subject {subject_number}...")

    # LSTM Model
    lstm_estimator = LSTMEstimator(data_folder, patient_id=f"subject{subject_number}")
    lstm_estimator.train_model(data_file, data_percentage=100)  # Vous pouvez ajuster le pourcentage de données ici

    # CNN Model
    cnn_estimator = CNNEstimator(data_folder, patient_id=f"subject{subject_number}")
    cnn_estimator.train_model(data_file, data_percentage=100)

    # RNN Model
    rnn_estimator = RNNEstimator(data_folder, patient_id=f"subject{subject_number}")
    rnn_estimator.train_model(data_file, data_percentage=100)

    # NRAX Model
    nrax_estimator = NRAXEstimator(data_folder, patient_id=f"subject{subject_number}")
    nrax_estimator.train_model(data_file, data_percentage=100)

    print(f"Training completed for subject {subject_number}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_models.py <subject_number>")
        sys.exit(1)

    subject_number = sys.argv[1]
    main(subject_number)
```

### Instructions pour exécuter le script :

1. **Sauvegardez le code ci-dessus** dans un fichier nommé `run_models.py` dans le même répertoire que vos autres fichiers de modèle.

2. **Assurez-vous que les fichiers de données** pour chaque sujet sont nommés correctement (par exemple, `subject1_labels.csv`, `subject2_labels.csv`, etc.) et qu'ils se trouvent dans le dossier `train_data_filtered_labeled_csv`.

3. **Exécutez le script** depuis la ligne de commande en fournissant le numéro du sujet comme argument. Par exemple :
   ```bash
   python run_models.py 2
   ```

Cela lancera l'entraînement de tous les modèles pour le sujet spécifié. Vous pouvez ajuster le pourcentage de données à utiliser pour l'entraînement en modifiant les appels à `train_model`.