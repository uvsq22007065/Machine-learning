import rosbag
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---- 1. Paramètres du Filtre Passe-bas ----
def lowpass_filter(data, cutoff=3, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    padlen = 3 * max(len(b), len(a))
    if data.shape[0] <= padlen:
        rospy.logwarn("[ERREUR] Données trop courtes pour le filtrage.")
        return data  # Retourne les données non filtrées si elles sont trop courtes

    return filtfilt(b, a, data, axis=0)

# ---- 2. Extraction des Données du Fichier .bag ----
def extract_data_from_bag(bag_file):
    bag = rosbag.Bag(bag_file)
    data_list = []

    for topic, msg, t in bag.read_messages():
        if hasattr(msg, 'data'):
            data = np.array(msg.data).reshape(-1, 16)  # Adapter à la structure du message
            data_list.append(data)

    bag.close()
    return np.concatenate(data_list, axis=0) if data_list else None

# ---- 3. Analyse des Données ----
def analyze_insole_data(data):
    data_filt = lowpass_filter(data)
    heel = np.sum(data_filt[:, 11:16], axis=1)
    mid = np.sum(data_filt[:, 5:11], axis=1)
    toe = np.sum(data_filt[:, 0:5], axis=1)

    heel[heel < 300] = mid[mid < 300] = toe[toe < 300] = 0
    return heel, mid, toe

# ---- 4. Calcul et Affichage du vGRF ----
def calculate_and_plot_vGRF(heel, mid, toe):
    vGRF = np.maximum(heel, np.maximum(mid, toe))

    if np.max(vGRF) == np.min(vGRF):
        rospy.logerr("[ERREUR] Données constantes. Normalisation impossible.")
        return

    vGRF_normalized = (vGRF - np.min(vGRF)) / (np.max(vGRF) - np.min(vGRF))

    plt.figure()
    plt.plot(vGRF_normalized, label='vGRF Normalized')
    plt.title('Normalized vGRF')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Force')
    plt.grid(True)
    plt.show()

# ---- 5. Main : Traitement du Fichier .bag ----
if __name__ == "__main__":
    rospy.init_node('bag_file_processor', anonymous=True)

    bag_file = "/data_bags/walking_treadmill_2d5kmph.bag"  # Chemin du fichier bag
    data = extract_data_from_bag(bag_file)

    if data is not None:
        heel, mid, toe = analyze_insole_data(data)
        calculate_and_plot_vGRF(heel, mid, toe)
    else:
        rospy.logerr("[ERREUR] Aucune donnée valide extraite du fichier .bag.")
