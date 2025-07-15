import os

def renomme_fichiers_csv_et_pkl(directory: str, old_prefix="RNNWS1stride16sequences", new_prefix="RNNWS"):
    """
    Parcourt récursivement tous les fichiers CSV et PKL et renomme ceux
    dont le nom commence par `old_prefix`, en le remplaçant par `new_prefix`.
    """
    renamed_count = 0

    for root, _, files in os.walk(directory):
        for file_name in files:
            if (file_name.__contains__(old_prefix)
                and (file_name.endswith('.csv') or file_name.endswith('.pkl'))):
                
                old_path = os.path.join(root, file_name)
                new_name = file_name.replace(old_prefix, new_prefix, 1)
                new_path = os.path.join(root, new_name)

                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"[Fichier] {old_path} → {new_path}")
                    renamed_count += 1
                else:
                    print(f"[Fichier] {new_path} existe déjà. Saut.")

    if renamed_count == 0:
        print("Aucun fichier .csv ou .pkl à renommer.")
    else:
        print(f"{renamed_count} fichiers renommés.")

if __name__ == "__main__":
    base_directory = "Reprise_24_mai_2025//results"
    renomme_fichiers_csv_et_pkl(base_directory)
