import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

# Fonction de filtrage adaptatif
def design_filter(data, fs):
    freqs, psd = signal.welch(data, fs=fs)
    median_power = np.median(psd)
    threshold = 5 * median_power
    try:
        cutoff_freq = freqs[np.where(psd > threshold)[0][0]]
        if cutoff_freq <= 0:
            cutoff_freq = 5
    except IndexError:
        cutoff_freq = 5
    b, a = signal.butter(2, cutoff_freq / (fs / 2), btype='low')
    return b, a

# Traitement du signal
def process_signal(data, fs):
    b, a = design_filter(data, fs)
    try:
        return signal.filtfilt(b, a, data, axis=0)
    except ValueError:
        return data

# Interpolation sur base temporelle commune
def interpolate_data(reference_time, target_time, target_data):
    return np.interp(reference_time, target_time, target_data)

# Calcul du gait vector robuste avec seuils fixes
def compute_fixed_threshold_gait_vector(vgrf_data, threshold_entry=0.5, threshold_exit=0.4, min_cycle_duration=30):
    gait_vector = np.zeros(len(vgrf_data))
    state = "waiting_for_contact"
    cycle_start = None

    i = 0
    while i < len(vgrf_data):
        force = vgrf_data[i]

        if state == "waiting_for_contact":
            if force > threshold_entry:
                cycle_start = i
                state = "in_stance"

        elif state == "in_stance":
            if force < threshold_exit:
                state = "in_swing"

        elif state == "in_swing":
            if force > threshold_entry:
                cycle_end = i
                if cycle_start is not None and (cycle_end - cycle_start) >= min_cycle_duration:
                    gait_vector[cycle_start:cycle_end] = np.linspace(0, 100, cycle_end - cycle_start)
                state = "in_stance"
                cycle_start = cycle_end
        i += 1

    return gait_vector

# Sauvegarde des CSV
def save_csv(data, folder, filename):
    data.to_csv(os.path.join(folder, filename), index=False)

# Visualisation des signaux
def plot_results(time, angle_data, vgrf_data, gait_progress):
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, angle_data, label='Ankle Angle')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, vgrf_data, label='vGRF', color='red')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, gait_progress, label='Gait Progress', color='green')
    plt.xlabel('Time')
    plt.ylabel('Progress (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Fonction principale
def process_files(folder):
    fs = 100  # Hz
    ankle_df = pd.read_csv(os.path.join(folder, "ankle_joint-angle.csv"))
    vgrf_df = pd.read_csv(os.path.join(folder, "vGRF.csv"))

    ankle_df["data"] = process_signal(ankle_df["data"].values, fs)
    vgrf_df["data"] = process_signal(vgrf_df["data"].values, fs)

    interpolated_vgrf = interpolate_data(ankle_df["Time"].values, vgrf_df["Time"].values, vgrf_df["data"].values)
    vgrf_df = pd.DataFrame({"Time": ankle_df["Time"], "data": interpolated_vgrf})

    gait_progress = compute_fixed_threshold_gait_vector(vgrf_df["data"].values)

    save_csv(pd.DataFrame({"Time": ankle_df["Time"], "Gait_Progress": gait_progress}), folder, "gait_progress.csv")
    save_csv(pd.DataFrame({"Time": ankle_df["Time"], "Gait_Vector": gait_progress}), folder, "gait_vector.csv")

    plot_results(ankle_df["Time"].values, ankle_df["data"].values, vgrf_df["data"].values, gait_progress)

    print("\nTraitement terminé et fichiers sauvegardés.")

# Point d'entrée
if __name__ == "__main__":
    folder_path = input("Entrez le chemin du dossier contenant les fichiers CSV : ")
    process_files(folder_path)
