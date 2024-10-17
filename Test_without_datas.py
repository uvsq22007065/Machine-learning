import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---- 1. Simulation du chargement des données ----
# Remplacez ceci par vos propres données lors de l'intégration.
np.random.seed(42)  # Pour des résultats reproductibles
insole_data = np.random.randint(0, 600, (1000, 16))  # Données simulées

# ---- 2. Filtrage Passe-bas ----
def lowpass_filter(data, cutoff=3, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

# Application du filtre sur les données brutes
insole_data_filt = lowpass_filter(insole_data)

# ---- 3. Division en Régions ----
heel = np.sum(insole_data_filt[:, 11:16], axis=1)
mid = np.sum(insole_data_filt[:, 5:11], axis=1)
toe = np.sum(insole_data_filt[:, 0:5], axis=1)

# Seuil : si la force est inférieure à 300, on la met à 0.
heel[heel < 300] = 0
mid[mid < 300] = 0
toe[toe < 300] = 0

# ---- 4. Estimation de la Force vGRF ----
vGRF_data = np.maximum(heel, np.maximum(mid, toe))

# Normalisation des forces entre 0 et 1
vGRF_normalized = (vGRF_data - np.min(vGRF_data)) / (np.max(vGRF_data) - np.min(vGRF_data))

# Tracé du vGRF normalisé
plt.figure()
plt.plot(vGRF_normalized)
plt.title('Normalized Vertical Ground Reaction Force (vGRF)')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Force')
plt.grid(True)
plt.show()

# ---- 5. Détection des Phases de Marche ----
def detect_gait_phases(heel, mid, toe):
    phases = []
    in_cycle_TO = False  # Cycle Toe-Off
    in_cycle_FF = False  # Cycle Flat-Foot

    for h, m, t in zip(heel, mid, toe):
        if h > 0.2 and (m < 0.1 or t < 0.1):
            phase = 'HS'  # Heel Strike
            in_cycle_TO = True  # Nouveau cycle Toe-Off détecté
            in_cycle_FF = True
        elif h < 0.1 and m < 0.1 and t < 0.1:
            phase = 'MSW'  # Mid-Swing
        elif h < 0.4 and m < 0.3 and t < 0.5 and in_cycle_TO:
            phase = 'TO'  # Toe-Off
            in_cycle_TO = False  # Cycle terminé
        elif m > 0.3 and h < 0.3 and t > 0.25:
            phase = 'HO'  # Heel-Off
        elif h > 0.25 and m > 0.25 and in_cycle_FF:
            phase = 'FF/MST'  # Flat-Foot/Mid-Stance
            in_cycle_FF = False  # Cycle terminé
        else:
            phase = 'Unknown'
        phases.append(phase)
    return phases

# Détection des phases de marche
gait_phases = detect_gait_phases(heel, mid, toe)

# Affichage des phases de marche
plt.figure()
plt.plot(vGRF_normalized, label='vGRF Normalized')
for i, phase in enumerate(gait_phases):
    if i % 100 == 0:  # Annoter chaque 100ème point pour éviter la surcharge
        plt.text(i, vGRF_normalized[i] + 0.05, phase, fontsize=8, ha='center')
plt.title('Gait Phases with Normalized vGRF')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Force')
plt.grid(True)
plt.show()
