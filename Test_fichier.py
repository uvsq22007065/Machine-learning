import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt, butter, filtfilt
from scipy.fft import fft
import scipy.io
import h5py 
import numpy as np

# Fonction pour convertir un fichier .mat en .npy
def convert_mat_to_npy(mat_file_path, npy_file_path):
    # Lire le fichier .mat
    with h5py.File(mat_file_path, 'r') as f:
        # Convertir les données en dictionnaire
        mat_data = {}
        for key in f.keys():
            # Si c'est un dataset, on lit les données
            if isinstance(f[key], h5py.Dataset):
                mat_data[key] = f[key][()]
            # Si c'est un groupe, on peut l'ajouter comme un groupe
            elif isinstance(f[key], h5py.Group):
                mat_data[key] = {sub_key: f[key][sub_key][()] for sub_key in f[key].keys()}

    # Sauvegarder le dictionnaire en .npy
    np.save(npy_file_path, mat_data)

# Chemins des fichiers .mat et .npy
file_1_mat_path = "data_bags/healthy_initial_posture_with_noise.mat"
file_1_npy_path = "data_bags/healthy_initial_posture_with_noise.npy"

file_2_mat_path = "data_bags/healthy_treadmill_noise2.mat"
file_2_npy_path = "data_bags/healthy_treadmill_noise2.npy"

# Conversion des fichiers
convert_mat_to_npy(file_1_mat_path, file_1_npy_path)
convert_mat_to_npy(file_2_mat_path, file_2_npy_path)

print("Conversion terminée avec succès !")

# ---- Chargement des données ----
initial_pos = np.load('data_bags/healthy_initial_posture_with_noise.npy', allow_pickle=True).item()
force_noise_calib = np.load('data_bags/healthy_treadmill_noise2.npy', allow_pickle=True).item()

# ---- Traitement des quaternions ----
# Convertir les quaternions en rotation pour le pied et le tibia
q_foot = R.from_quat(np.column_stack((
    initial_pos['imu_data_foot']['QuatW'],
    initial_pos['imu_data_foot']['QuatX'],
    initial_pos['imu_data_foot']['QuatY'],
    initial_pos['imu_data_foot']['QuatZ']
)))

q_shank = R.from_quat(np.column_stack((
    initial_pos['imu_data_shank']['QuatW'],
    initial_pos['imu_data_shank']['QuatX'],
    initial_pos['imu_data_shank']['QuatY'],
    initial_pos['imu_data_shank']['QuatZ']
)))

# Interpolation des quaternions du pied avec les timestamps du tibia
t_foot = initial_pos['imu_data_foot']['TimeStampGlob']
t_shank = initial_pos['imu_data_shank']['TimeStampGlob']

interp_func = interp1d(t_foot, np.column_stack((
    initial_pos['imu_data_foot']['QuatW'],
    initial_pos['imu_data_foot']['QuatX'],
    initial_pos['imu_data_foot']['QuatY'],
    initial_pos['imu_data_foot']['QuatZ']
)), axis=0)

q_foot_interp = R.from_quat(interp_func(t_shank))

# ---- Calcul des angles de cheville ----
ankle_angles = [q_foot_interp[i].inv() * q_shank[i] for i in range(len(q_shank))]
ankle_angles_deg = np.array([angle.magnitude() for angle in ankle_angles])  # Conversion en degrés

# ---- Filtrage des angles de cheville ----
ankle_angles_deg = np.nan_to_num(ankle_angles_deg)  # Remplacer NaN par 0
ankle_angles_filt = medfilt(ankle_angles_deg, kernel_size=5)  # Filtre médian

# Compensation d'angle
compensator = np.deg2rad(9)  # Compensateur de 9 degrés
initial_angle = np.median(ankle_angles_filt[5:-5]) + compensator
ankle_angles_filt -= initial_angle  # Appliquer la compensation

# ---- Tracé des angles de cheville ----
plt.figure()
plt.plot(t_shank, ankle_angles_deg, label="Angles de cheville bruts")
plt.plot(t_shank, ankle_angles_filt, label="Angles de cheville filtrés", linestyle='--')
plt.xlabel("Temps (s)")
plt.ylabel("Angle (degrés)")
plt.title("Angles de cheville")
plt.legend()
plt.grid(True)
plt.show()

# ---- Analyse FFT sur les données de force (vGRF) ----
Fs = 100  # Fréquence d'échantillonnage en Hz
force_data = force_noise_calib['insole_data']['Data'][:, 0]  # Utilisation d'un capteur pour l'exemple
N = len(force_data)

# Transformation de Fourier
Y = fft(force_data)
f = np.linspace(0, Fs, N)

# Amplitude du signal
P2 = np.abs(Y / N)
P1 = P2[:N // 2 + 1]
P1[1:-1] *= 2

# Affichage du spectre de fréquence pour choisir la fréquence de coupure
plt.figure()
plt.plot(f[:N // 2 + 1], P1)
plt.title('FFT des données de force')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# ---- Filtrage des données de force ----
cutoff_freq = 5  # Exemple de fréquence de coupure à 5 Hz
b, a = butter(N=4, Wn=cutoff_freq / (Fs / 2), btype='low', analog=False)
filtered_force_data = filtfilt(b, a, force_data)

# ---- Détection des phases de marche ----
threshold = np.mean(filtered_force_data)  # Seuil simple basé sur la moyenne
gait_phases = filtered_force_data > threshold  # Détection des phases

# ---- Tracé des phases de marche ----
plt.figure()
plt.plot(filtered_force_data, label="Force filtrée")
plt.plot(gait_phases * np.max(filtered_force_data), label="Phases de marche", linestyle='--')
plt.legend()
plt.title('Détection des phases de marche')
plt.grid(True)
plt.show()

# ---- Sorties finales ----
force_data_train = filtered_force_data
gait_vector = gait_phases
ankle_angles_filt = ankle_angles_filt
