import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Matrice ajustée
confusion_matrix = np.array([
    [622, 52, 76, 4],   # FF/MST
    [72, 906, 87, 0],    # HO
    [64, 61, 1071, 0],    # HS
    [0, 1, 0, 1480]     # MSW
])

# Calcul du total des prédictions
total_predictions = np.sum(confusion_matrix)

# Calcul du total des prédictions correctes (diagonale)
correct_predictions = np.trace(confusion_matrix)

# Calcul du pourcentage de validation
accuracy_percentage = (correct_predictions / total_predictions) * 100

# Affichage
print(f"Pourcentage de validation : {accuracy_percentage:.2f} %")

# Classes réelles et prédites
classes = ['FF/MST', 'HO', 'HS', 'MSW']

# Création du graphique
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

# Ajouter des labels et un titre
plt.title('Confusion matrix')
plt.xlabel('Predicted classes')
plt.ylabel('Real classes')

# Affichage
plt.show()
