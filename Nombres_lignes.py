import os
import time
import re
from collections import Counter

def count_code_statistics(directory):
    start_time = time.time()
    
    total_lines = 0
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    function_count = 0
    class_count = 0
    file_count = 0
    unique_lines = set()
    all_code_lines = []
    all_characters = []

    # Caractères
    total_characters = 0
    code_characters = 0
    comment_characters = 0
    whitespace_characters = 0

    print(f"Analyse du dossier : {directory}\n")

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".py", ".m")):
                file_count += 1
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        inside_function = False

                        for line in lines:
                            total_characters += len(line)
                            whitespace_characters += sum(1 for c in line if c in (' ', '\t'))

                            stripped = line.strip()

                            if not stripped:
                                blank_lines += 1
                                continue

                            if stripped.startswith('#') or stripped.startswith('%'):
                                comment_lines += 1
                                comment_characters += len(line)
                                continue

                            # Lignes de code utiles
                            code_lines += 1
                            code_characters += len(line)
                            unique_lines.add(stripped)
                            all_code_lines.append(stripped)
                            all_characters.extend(line)

                            # Détection de fonctions/classes
                            if stripped.startswith("def ") or stripped.startswith("function"):
                                function_count += 1
                            if stripped.startswith("class "):
                                class_count += 1

                except Exception as e:
                    print(f"Erreur lors de la lecture de {filepath} : {e}")

    # Calculs finaux
    average_lines_per_file = total_lines / file_count if file_count else 0
    average_characters_per_code_line = code_characters / code_lines if code_lines else 0
    most_common_lines = Counter(all_code_lines).most_common(10)
    most_common_chars = Counter(all_characters).most_common(10)

    duration = time.time() - start_time

    print("\n----- Résultat -----")
    print(f"📄 Fichiers analysés (.py et .m)                         : {file_count}")
    print(f"📏 Nombre total de lignes                               : {total_lines}")
    print(f"🧾 Lignes de code utiles                                : {code_lines}")
    print(f"💬 Lignes de commentaires                               : {comment_lines}")
    print(f"⬜ Lignes vides                                         : {blank_lines}")
    print(f"🔁 Lignes de code uniques                               : {len(unique_lines)}")
    print(f"🔧 Fonctions détectées                                  : {function_count}")
    print(f"🏷️  Classes détectées (Python uniquement)               : {class_count}")
    print(f"📊 Moyenne de lignes par fichier                        : {average_lines_per_file:.2f}")
    print(f"🔡 Nombre total de caractères                           : {total_characters}")
    print(f"🔢 Caractères dans les lignes de code                   : {code_characters}")
    print(f"🗨️  Caractères dans les commentaires                    : {comment_characters}")
    print(f"⬜ Caractères d'espacement (espaces, tabulations)       : {whitespace_characters}")
    print(f"📈 Moyenne de caractères par ligne de code utile        : {average_characters_per_code_line:.2f}")
    print(f"⏱️  Temps total d'exécution                             : {duration:.2f} secondes")

    print("\n🧠 Top 10 lignes de code les plus fréquentes :")
    for i, (line, count) in enumerate(most_common_lines, 1):
        print(f"  {i:2}. ({count}x) {line}")

    print("\n🔤 Top 10 caractères les plus fréquents :")
    for i, (char, count) in enumerate(most_common_chars, 1):
        display = repr(char) if char.strip() == '' else char
        print(f"  {i:2}. '{display}' → {count} fois")

# === UTILISATION ===
count_code_statistics("C:/Users/Grégoire/OneDrive/Bureau/EPF/BRL/Machine learning")
