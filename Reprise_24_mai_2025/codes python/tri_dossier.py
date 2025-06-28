import os
import re
import shutil

results_dir = "Reprise_24_mai_2025\\results"

# Parcours des dossiers dans results
for entry in os.listdir(results_dir):
    entry_path = os.path.join(results_dir, entry)
    if os.path.isdir(entry_path):
        # Recherche du numéro du subject dans le nom du dossier
        match = re.search(r"subject(\d+)", entry, re.IGNORECASE)
        if match:
            subject_num = match.group(1)
            target_dir = os.path.join(results_dir, f"results_subject{subject_num}")
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(entry_path, os.path.join(target_dir, entry))