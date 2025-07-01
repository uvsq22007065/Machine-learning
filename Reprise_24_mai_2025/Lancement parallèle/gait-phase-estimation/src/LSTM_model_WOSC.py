import os
import sys

def run_model(model_name, subject_id):
    """
    Function to run a specific model based on the model name and subject ID.
    """
    model_script = f"{model_name}_model_WOSC.py"
    if os.path.exists(model_script):
        os.system(f"python {model_script} {subject_id}")
    else:
        print(f"Model script {model_script} does not exist.")

def main(subject_id):
    models = ['RNN', 'CNN', 'NRAX', 'LSTM']  # List of models to run
    for model in models:
        print(f"Running {model} model for subject {subject_id}...")
        run_model(model, subject_id)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_all_models.py <subject_id>")
        sys.exit(1)

    subject_id = sys.argv[1]
    main(subject_id)