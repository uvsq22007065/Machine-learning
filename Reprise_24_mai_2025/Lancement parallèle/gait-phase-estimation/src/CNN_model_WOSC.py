import os
import sys

def run_model(model_name, subject_id):
    # Construct the command to run the model
    command = f"python {model_name} {subject_id}"
    os.system(command)

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_models.py <subject_id>")
        sys.exit(1)

    subject_id = sys.argv[1]

    # List of model scripts to run
    models = [
        "RNN_model_WOSC.py",
        "CNN_model_WOSC.py",
        "NRAX_model_WOSC.py",
        "LSTM_model_WOSC.py"
    ]

    for model in models:
        print(f"Running {model} for subject {subject_id}...")
        run_model(model, subject_id)

if __name__ == "__main__":
    main()