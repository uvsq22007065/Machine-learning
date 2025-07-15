import os

def renomme_dossiers_et_fichiers(directory: str, old_prefix="RNNWS1stride16sequences", new_prefix="RNNWS"):
    """
    Parcourt récursivement tous les fichiers et dossiers à partir de `directory` et
    renomme tous ceux qui commencent par `old_prefix` par `new_prefix`.
    Répète le processus jusqu'à ce qu'il n'y ait plus rien à renommer.
    """
    changements = True
    iteration = 1

    while changements:
        print(f"\n--- Itération {iteration} ---")
        changements = False

        for root, dirs, files in os.walk(directory, topdown=False):
            # Dossiers
            for dir_name in dirs:
                if dir_name.__contains__(old_prefix):
                    old_path = os.path.join(root, dir_name)
                    new_name = dir_name.replace(old_prefix, new_prefix, 1)
                    new_path = os.path.join(root, new_name)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"[Dossier] {old_path} → {new_path}")
                        changements = True
                    else:
                        print(f"[Dossier] {new_path} existe déjà. Saut.")

            # Fichiers
            for file_name in files:
                if file_name.startswith(old_prefix):
                    old_path = os.path.join(root, file_name)
                    new_name = file_name.replace(old_prefix, new_prefix, 1)
                    new_path = os.path.join(root, new_name)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"[Fichier] {old_path} → {new_path}")
                        changements = True
                    else:
                        print(f"[Fichier] {new_path} existe déjà. Saut.")

        iteration += 1

    print("\nRenommage terminé.")

if __name__ == "__main__":
    base_directory = "Reprise_24_mai_2025//results"
    renomme_dossiers_et_fichiers(base_directory)
