import os
import glob
import pandas as pd
import numpy as np

def analyze_subject(subject_number):
    print(f"\nComparing data between algorithms for subject{subject_number}:")
    pattern = os.path.join(folder_path, f"subject{subject_number}_labels*")
    files = glob.glob(pattern)
    dataframes = {}

    for file in files:
        if "ALGORITHM" not in file:
            basename = os.path.basename(file)
            algorithm = basename.replace(f"subject{subject_number}_labels", "").replace(".xlsx", "")
            try:
                df = pd.read_excel(file)
                dataframes[algorithm] = df
                print(f"\nLoaded {basename} with {len(df)} rows")
            except Exception as e:
                print(f"Error reading file: {e}")

    if len(dataframes) > 1:
        print(f"\nComparing data between algorithms for subject {subject_number}:")
        algorithms = list(dataframes.keys())
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                algo1, algo2 = algorithms[i], algorithms[j]
                df1, df2 = dataframes[algo1], dataframes[algo2]
                
                if df1.equals(df2):
                    print(f"{algo1} and {algo2}: IDENTICAL DATA")
                else:
                    if len(df1) != len(df2):
                        print(f"{algo1} and {algo2}: DIFFERENT NUMBER OF ROWS ({len(df1)} vs {len(df2)})")
                    else:
                        differences = (df1 != df2).sum().sum()
                        print(f"{algo1} and {algo2}: {differences} different values found")

# Chemin du dossier contenant les fichiers de labels
folder_path = os.path.dirname(os.path.abspath(__file__))

# Analyser chaque sujet
for subject in range(1, 5):
    analyze_subject(subject)
