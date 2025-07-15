import os
import shutil

def move_classical_folders(base_path: str, destination_folder_name: str = "classical_WOL", num_subjects: int = 11):
    destination_path = os.path.join(base_path, destination_folder_name)
    os.makedirs(destination_path, exist_ok=True)
    results_base = os.path.join(base_path, "results")  # Le dossier parent des results_subjectX

    for subj_idx in range(1, num_subjects + 1):
        subject_folder = os.path.join(results_base, f"results_subject{subj_idx}")
        if not os.path.isdir(subject_folder):
            print(f"Attention : {subject_folder} n'existe pas, passage au suivant.")
            continue

        for subfolder in os.listdir(subject_folder):
            if subfolder.startswith("Classical_subject"):
                source_path = os.path.join(subject_folder, subfolder)
                dest_path = os.path.join(destination_path, subfolder)

                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(subfolder)
                    count = 1
                    new_subfolder = f"{base}_{count}{ext}"
                    new_dest_path = os.path.join(destination_path, new_subfolder)
                    while os.path.exists(new_dest_path):
                        count += 1
                        new_subfolder = f"{base}_{count}{ext}"
                        new_dest_path = os.path.join(destination_path, new_subfolder)
                    dest_path = new_dest_path

                print(f"Déplacement de {source_path} vers {dest_path}")
                shutil.move(source_path, dest_path)

if __name__ == "__main__":
    base_path = "Reprise_24_mai_2025"
    move_classical_folders(base_path)
