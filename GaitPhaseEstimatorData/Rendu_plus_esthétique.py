import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    
    # Étapes du pipeline
    steps = [
        ("Input Data", 1, "Forces, Angles, Derivatives"),
        ("Preprocessing", 3, "Standardization, Mapping"),
        ("ML Models", 5, "GB, RF, Voting"),
        ("Prediction", 7, "Gait Progress %")
    ]
    
    # Dessiner les boîtes
    for step, x, desc in steps:
        box = FancyBboxPatch((x, 2.5), 2, 1, boxstyle="round,pad=0.2", edgecolor='black', facecolor='#7aa6c2')
        ax.add_patch(box)
        ax.text(x + 1, 3, step, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x + 1, 2.7, desc, ha='center', va='center', fontsize=10)
    
    # Flèches de connexion
    for i in range(len(steps) - 1):
        x_start = steps[i][1] + 2
        x_end = steps[i + 1][1]
        ax.arrow(x_start, 3, x_end - x_start - 0.2, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Ajouter la représentation du cycle de marche
    gait_colors = ['#1f3b6f', '#1e6091', '#168aad', '#76c893', '#adb5bd', '#adb5bd', '#b5179e', '#7209b7', '#560bad']
    for i, color in enumerate(gait_colors):
        ax.add_patch(FancyBboxPatch((i + 0.5, 0.5), 0.9, 0.9, boxstyle="square", facecolor=color))
    
    ax.text(4.5, 0.2, "Gait Cycle Representation", ha='center', fontsize=12, fontweight='bold')
    plt.show()

draw_pipeline()
