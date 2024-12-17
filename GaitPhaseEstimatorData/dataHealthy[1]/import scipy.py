import scipy.io

# Charger le fichier .mat
mat_file_path = 'C:\\Users\\Grégoire\\OneDrive\\Bureau\\EPF\\BRL\\Machine learning\\GaitPhaseEstimatorData\\dataHealthy[1]\\Subject1_left.mat'
mat_data = scipy.io.loadmat(mat_file_path)

# Vérifier les clés principales
keys = ['vGRF', 'angle', 'torque']
for key in keys:
    if key in mat_data:
        print(f"--- {key} ---")
        data = mat_data[key]
        print(f"Type: {type(data)}")
        print(f"Shape: {data.shape if hasattr(data, 'shape') else 'Not an array'}")
        print(f"Content sample: {data[:5] if hasattr(data, '__getitem__') else 'Non-indexable'}\n")
    else:
        print(f"{key} not found in the .mat file.\n")
