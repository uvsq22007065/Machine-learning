import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as R

# Chemin des données
data_bags_path = "csv_files/"

# Fichiers de données mis à jour
filenames_2kmph = [
    "healthy_walking_treadmill_2kmph_filtered_foot.csv", 
    "healthy_walking_treadmill_2kmph_filtered_insole.csv", 
    "healthy_walking_treadmill_2kmph_filtered_shank.csv", 
    "healthy_walking_treadmill_2kmph_imu_data_foot.csv", 
    "healthy_walking_treadmill_2kmph_imu_data_shank.csv", 
    "healthy_walking_treadmill_2kmph_insole_data.csv"
]

filenames_2kmph2 = [
    "healthy_walking_treadmill_2kmph2_filtered_foot.csv", 
    "healthy_walking_treadmill_2kmph2_filtered_insole.csv", 
    "healthy_walking_treadmill_2kmph2_filtered_shank.csv", 
    "healthy_walking_treadmill_2kmph2_imu_data_foot.csv", 
    "healthy_walking_treadmill_2kmph2_imu_data_shank.csv", 
    "healthy_walking_treadmill_2kmph2_insole_data.csv"
]

filenames_2d5kmph = [
    "healthy_walking_treadmill_2d5kmph_filtered_foot.csv", 
    "healthy_walking_treadmill_2d5kmph_filtered_insole.csv", 
    "healthy_walking_treadmill_2d5kmph_filtered_shank.csv", 
    "healthy_walking_treadmill_2d5kmph_imu_data_foot.csv", 
    "healthy_walking_treadmill_2d5kmph_imu_data_shank.csv", 
    "healthy_walking_treadmill_2d5kmph_insole_data.csv"
]

filenames_initial_posture = [
    "healthy_initial_posture_with_noise_filtered_foot.csv", 
    "healthy_initial_posture_with_noise_filtered_insole.csv", 
    "healthy_initial_posture_with_noise_filtered_shank.csv", 
    "healthy_initial_posture_with_noise_imu_data_foot.csv", 
    "healthy_initial_posture_with_noise_imu_data_shank.csv", 
    "healthy_initial_posture_with_noise_insole_data.csv"
]

filename_force_noise = [
    "healthy_treadmill_noise2_insole_data.csv", 
    "healthy_treadmill_noise2_imu_data_topic.csv", 
    "healthy_treadmill_noise2_imu_data_msgs.csv"
]

filename_insole_points = [
    "healthy_determine_points_insole_data.csv", 
    "healthy_determine_points_imu_data_topic.csv", 
    "healthy_determine_points_imu_data_msgs.csv"
]

# Chargement des données
initial_pos_foot = pd.read_csv(data_bags_path + filenames_initial_posture[3])
initial_pos_shank = pd.read_csv(data_bags_path + filenames_initial_posture[4])
force_noise_calib = pd.read_csv(data_bags_path + filename_force_noise[0])
insole_points = pd.read_csv(data_bags_path + filename_insole_points[0])
healthy_data = pd.read_csv("healthy_ankle_angle.txt", delim_whitespace=True, header=None)

# Initial Posture
initial_pos_foot['t_foot'] = initial_pos_foot['TimeStampGlob']
initial_pos_shank['t_shank'] = initial_pos_shank['TimeStampGlob']

# Gestion des quaternions avec scipy Rotation
def quaternion(quat_data):
    return R.from_quat(quat_data)

# Remplacer par une fonction pour calculer les quaternions
initial_pos_foot['q_foot'] = quaternion(initial_pos_foot[['QuatW', 'QuatX', 'QuatY', 'QuatZ']].values)
initial_pos_foot['q_foot_interp'] = quaternion(
    interp1d(initial_pos_foot['t_foot'], 
             initial_pos_foot[['QuatW', 'QuatX', 'QuatY', 'QuatZ']].values, 
             axis=0, fill_value="extrapolate")(initial_pos_shank['t_shank'])
)
initial_pos_shank['q_shank'] = quaternion(initial_pos_shank[['QuatW', 'QuatX', 'QuatY', 'QuatZ']].values)

# Calculer les angles de la cheville
def dist(q1, q2):
    # Calcul de la distance angulaire entre deux quaternions
    return R.from_quat(q1).inv() * R.from_quat(q2).magnitude()

initial_pos_shank['ankle_angles'] = np.array([
    dist(initial_pos_foot['q_foot_interp'][i], initial_pos_shank['q_shank'][i]) 
    for i in range(len(initial_pos_shank['q_shank']))
])
initial_pos_shank['ankle_angles'][np.isnan(initial_pos_shank['ankle_angles'])] = 0
initial_pos_shank['ankle_angles_filt'] = pd.Series(initial_pos_shank['ankle_angles']).rolling(window=10, center=True).median()

# Compensation angulaire
compensator = 0.16  # 9 degrees
initial_angle = np.median(initial_pos_shank['ankle_angles_filt'][5:-5]) + compensator

# Angle Estimator
t_foot = initial_pos_foot['TimeStampGlob']
t_shank = initial_pos_shank['TimeStampGlob']

q_foot = quaternion(initial_pos_foot[['QuatW', 'QuatX', 'QuatY', 'QuatZ']].values)
q_shank = quaternion(initial_pos_shank[['QuatW', 'QuatX', 'QuatY', 'QuatZ']].values)

q_foot_interp = quaternion(
    interp1d(t_foot, q_foot.as_quat(), axis=0, fill_value="extrapolate")(t_shank)
)

ankle_angles = np.zeros(len(q_shank))

for i in range(len(q_shank)):
    ankle_angles[i] = dist(q_foot_interp[i], q_shank[i]) - initial_angle

ankle_angles[np.isnan(ankle_angles)] = 0
ankle_angles_filt = pd.Series(ankle_angles).rolling(window=10, center=True).median()

# Affichage des résultats
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(ankle_angles, label='Angles de la cheville')
plt.plot(ankle_angles_filt, label='Angles filtrés de la cheville')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(healthy_data[0], healthy_data[1])
plt.show()

# Filtrage des données vGRF Analyzer
cutoff_freq = 3
Fs = 100
b, a = butter(4, cutoff_freq / (0.5 * Fs), btype='low')

# Appliquer le filtre aux données de calibration du bruit
for i in range(force_noise_calib.shape[1]):
    filtered_data = lfilter(b, a, force_noise_calib.iloc[:, i].astype(float))
    force_noise_calib[f'DataFilt_{i}'] = filtered_data

# Visualisation des résultats filtrés
plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(force_noise_calib.iloc[:, :].values)  # Original Signal Calibration
plt.title('Signal Original Calibration')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(force_noise_calib.filter(like='DataFilt').values)  # Filtered Signal Calibration
plt.title('Signal Filtré Calibration')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

# Estimation de la force vGRF
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

# Détermination des phases de la démarche
gait_phases = []
cycle_starts = []
phase_change_indices = []
phase_labels = []

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

    if true_heel_force > 0.2 and (true_mid_force < 0.1 or true_toe_force < 0.1):
        phase = 'HS'
        if len(cycle_starts) == 0 or (i - cycle_starts[-1]) > 100:
            cycle_starts.append(i)
    elif true_mid_force > 0.2 and true_heel_force < 0.1 and true_toe_force < 0.1:
        phase = 'MS'
    elif true_toe_force > 0.2 and true_mid_force < 0.1 and true_heel_force < 0.1:
        phase = 'TO'
    else:
        phase = 'Swing'
    
    gait_phases.append(phase)
    phase_change_indices.append(i)
    phase_labels.append(phase)
