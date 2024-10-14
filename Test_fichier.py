import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter

# Chargement des données
data_bags_path = "csv_files/"
filename = "healthy_walking_treadmill_2kmph.csv"
filename2 = "healthy_walking_treadmill_2kmph2.csv"
filename3 = "healthy_walking_treadmill_2d5kmph.csv"
filename_initial_pos = "healthy_initial_posture_with_noise.csv"
filename_force_noise = "healthy_treadmill_noise2.csv"
filename_insole_points = "healthy_determine_points.csv"

# Chargement des données
initial_pos = pd.read_csv(data_bags_path + filename_initial_pos)
force_noise_calib = pd.read_csv(data_bags_path + filename_force_noise)
insole_points = pd.read_csv(data_bags_path + filename_insole_points)
healthy_data = pd.read_csv("healthy_ankle_angle.txt", delim_whitespace=True, header=None)

# Initial Posture
initial_pos['t_foot'] = initial_pos['imu_data_foot.TimeStampGlob']
initial_pos['t_shank'] = initial_pos['imu_data_shank.TimeStampGlob']
# Remplacer par une fonction pour calculer les quaternions
initial_pos['q_foot'] = quaternion(initial_pos[['imu_data_foot.QuatW', 'imu_data_foot.QuatX', 
                                                 'imu_data_foot.QuatY', 'imu_data_foot.QuatZ']].values)
initial_pos['q_foot_interp'] = quaternion(interp1d(initial_pos['t_foot'], 
                                       initial_pos[['imu_data_foot.QuatW', 'imu_data_foot.QuatX', 
                                       'imu_data_foot.QuatY', 'imu_data_foot.QuatZ']].values, 
                                       fill_value="extrapolate")(initial_pos['t_shank']))
initial_pos['q_shank'] = quaternion(initial_pos[['imu_data_shank.QuatW', 'imu_data_shank.QuatX', 
                                                 'imu_data_shank.QuatY', 'imu_data_shank.QuatZ']].values)

initial_pos['ankle_angles'] = np.array([dist(initial_pos['q_foot_interp'][i], 
                                              initial_pos['q_shank'][i]) for i in range(len(initial_pos['q_shank']))])
initial_pos['ankle_angles'][np.isnan(initial_pos['ankle_angles'])] = 0
initial_pos['ankle_angles_filt'] = smoothdata(initial_pos['ankle_angles'], method='movmedian', 
                                               smoothing_factor=0.1)

compensator = 0.16  # 9 degrees
initial_angle = np.median(initial_pos['ankle_angles_filt'][5:-5]) + compensator

# Angle Estimator
# Synchronisation des données des IMUs (pied et tibia) et interpolation
t_foot = initial_pos['imu_data_foot.TimeStampGlob']
t_shank = initial_pos['imu_data_shank.TimeStampGlob']

# Création des quaternions pour le pied et le tibia
q_foot = quaternion(initial_pos[['imu_data_foot.QuatW', 'imu_data_foot.QuatX', 
                                 'imu_data_foot.QuatY', 'imu_data_foot.QuatZ']].values)
q_shank = quaternion(initial_pos[['imu_data_shank.QuatW', 'imu_data_shank.QuatX', 
                                   'imu_data_shank.QuatY', 'imu_data_shank.QuatZ']].values)

# Interpolation des quaternions du pied sur la même échelle temporelle que le tibia
q_foot_interp = quaternion(interp1d(t_foot, q_foot, fill_value="extrapolate")(t_shank))

# Initialisation de l'angle de référence (initial_angle)
ankle_angles = np.zeros(len(q_shank))

# Calcul des angles entre le pied et le tibia
for i in range(len(q_shank)):
    ankle_angles[i] = dist(q_foot_interp[i], q_shank[i]) - initial_angle

# Remplacement des valeurs NaN par 0
ankle_angles[np.isnan(ankle_angles)] = 0

# Filtrage des angles avec un filtre de médiane mobile
ankle_angles_filt = smoothdata(ankle_angles, method='movmedian', smoothing_factor=0.1)

# Affichage des résultats
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(ankle_angles, label='Angles de la cheville')
plt.plot(ankle_angles_filt, label='Angles filtrés de la cheville')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(healthy_data[0], healthy_data[1])
plt.show()

# vGRF Analyzer to determine Cutoff Frequency
cutoff_freq = 3
Fs = 100

# Low-Pass Filter Design
b, a = butter(4, cutoff_freq / (0.5 * Fs), btype='low')

# Appliquer le filtre aux données
for i in range(force_noise_calib.shape[1]):
    filtered_data = lfilter(b, a, force_noise_calib.iloc[:, i].astype(float))
    force_noise_calib[f'DataFilt_{i}'] = filtered_data

# Visualisation des résultats filtrés
plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(force_noise_calib.iloc[:, :])  # Original Signal Calibration
plt.title('Signal Original Calibration')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(force_noise_calib.filter(like='DataFilt'))  # Filtered Signal Calibration
plt.title('Signal Filtré Calibration')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

# vGRF Estimator
heel = np.sum(force_noise_calib.iloc[:, 12:17], axis=1)
heel[heel < 300] = 0
mid = np.sum(force_noise_calib.iloc[:, 6:12], axis=1)
mid[mid < 300] = 0
tip = np.sum(force_noise_calib.iloc[:, :5], axis=1)
tip[tip < 300] = 0

vGRF_data = np.maximum(heel, np.maximum(mid, tip))
vGRF_normalized = (vGRF_data - np.min(vGRF_data)) / (np.max(vGRF_data) - np.min(vGRF_data))

plt.figure(3)
plt.plot(vGRF_normalized)
plt.title('Force de Réaction au Sol Vertical Normalisée (vGRF)')
plt.xlabel('Time (s)')
plt.ylabel('Force Normalisée')
plt.grid()

# Determine Gait Phases
# Initialisation des variables
gait_phases = []
cycle_starts = []
phase_change_indices = []
phase_labels = []

# Calibration
known_force = 1
total_force = np.zeros(len(heel))

for i in range(len(heel)):
    if (heel[i] > 333) or (mid[i] > 333) or (tip[i] > 333):
        total_force[i] = heel[i] + mid[i] + tip[i]
        true_heel_force = known_force * heel[i] / total_force[i]
        true_mid_force = known_force * mid[i] / total_force[i]
        true_toe_force = known_force * tip[i] / total_force[i]
    else:
        true_heel_force = 0
        true_mid_force = 0
        true_toe_force = 0

    # Détection des phases de marche en fonction des conditions sur les forces
    if true_heel_force > 0.2 and (true_mid_force < 0.1 or true_toe_force < 0.1):
        phase = 'HS'
        if len(cycle_starts) == 0 or (i - cycle_starts[-1]) > 100:
            cycle_starts.append(i)
    elif true_heel_force < 0.1 and true_mid_force < 0.1 and true_toe_force < 0.1:
        phase = 'MSW'
    elif true_heel_force < 0.4 and true_mid_force < 0.3 and true_toe_force < 0.5:
        phase = 'TO'
    elif true_mid_force > 0.3 and true_heel_force < 0.3 and true_toe_force > 0.25:
        phase = 'HO'
    elif true_heel_force > 0.25 and true_mid_force > 0.25:
        phase = 'FF/MST'

    gait_phases.append(phase)

    # Détecter les changements de phase
    if i == 0 or gait_phases[i] != gait_phases[i - 1]:
        phase_change_indices.append(i)
        phase_labels.append(phase)

# Output des données de force et des phases de marche
np.savez('output_data.npz', force_data=true_total_forces, gait_phases=gait_phases, 
         phase_change_indices=phase_change_indices, phase_labels=phase_labels)
