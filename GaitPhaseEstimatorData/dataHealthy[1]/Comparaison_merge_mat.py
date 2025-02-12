import scipy.io
import numpy as np
import os
from itertools import combinations

import scipy.io
import numpy as np
import os
from itertools import combinations

def merge_files_if_compatible(results, output_dir, thresholds={'vGRF': 0.95, 'angle': 0.95}):
    """Merge an unlimited number of compatible .mat files by concatenating their data, with filenames showing metrics compatibility."""
    # Create a mapping of file clusters
    clusters = []
    file_to_cluster = {}
    incompatibility_log = []

    for result in results:
        file1 = result['file1']
        file2 = result['file2']

        # Check compatibility for each metric and calculate overall compatibility
        metrics = result['metrics']
        individual_compatibility = {
            key: metrics[key]['correlation'] >= thresholds[key] for key in metrics
        }
        average_correlation = np.mean([metrics[key]['correlation'] for key in metrics])
        overall_compatible = average_correlation >= 0.9  # Adjustable overall threshold

        # Log incompatibility reasons
        if not all(individual_compatibility.values()):
            incompatibility_log.append(
                f"Incompatible: {file1} <-> {file2} | "
                f"Metrics: {individual_compatibility}, Avg Correlation: {average_correlation:.2f}"
            )

        if overall_compatible:
            # Find existing clusters
            cluster1 = file_to_cluster.get(file1)
            cluster2 = file_to_cluster.get(file2)

            if cluster1 and cluster2 and cluster1 != cluster2:
                # Merge two clusters
                cluster1.update(cluster2)
                for file in cluster2:
                    file_to_cluster[file] = cluster1
                clusters.remove(cluster2)
            elif cluster1:
                cluster1.add(file2)
                file_to_cluster[file2] = cluster1
            elif cluster2:
                cluster2.add(file1)
                file_to_cluster[file1] = cluster2
            else:
                # Create a new cluster
                new_cluster = set([file1, file2])
                clusters.append(new_cluster)
                file_to_cluster[file1] = new_cluster
                file_to_cluster[file2] = new_cluster

    merged_files = []

    for cluster in clusters:
        # Load and concatenate data from all files in the cluster
        merged_data = {}
        compatible_metrics = set()
        for file in cluster:
            data = load_mat_file(file)
            if data is None:
                continue

            for key, value in data.items():
                if key not in merged_data:
                    merged_data[key] = value
                else:
                    merged_data[key] = np.concatenate((merged_data[key], value))

            # Track compatible metrics
            for key, is_compatible in individual_compatibility.items():
                if is_compatible:
                    compatible_metrics.add(key)

        # Generate a merged file name
        metrics_str = "_".join(sorted(compatible_metrics))
        merged_filename = f"merged_{metrics_str}_" + "_".join(
            [os.path.basename(file).replace('.mat', '') for file in cluster]
        ) + ".mat"
        merged_filepath = os.path.join(output_dir, merged_filename)

        # Save the merged data to a new .mat file
        scipy.io.savemat(merged_filepath, merged_data)
        merged_files.append(merged_filepath)

        print(f"Merged: {', '.join(cluster)} -> {merged_filepath}")

    # Output incompatibility log
    print("\nIncompatibility log:")
    for log in incompatibility_log:
        print(log)

    return merged_files

def compare_signals(signal1, signal2):
    """Compare two signals and return similarity metrics."""
    min_length = min(len(signal1), len(signal2))
    signal1, signal2 = signal1[:min_length], signal2[:min_length]
    
    # Calculate similarity metrics
    mse = np.mean((signal1 - signal2) ** 2)
    correlation = np.corrcoef(signal1, signal2)[0, 1]
    return mse, correlation

def load_mat_file(file_path):
    """Load .mat file and return relevant keys."""
    try:
        mat_data = scipy.io.loadmat(file_path)
        return {
            'vGRF': mat_data['vGRF'][0, 0][0][0],  # Ground Reaction Force
            'angle': mat_data['angle'][0, 0][0][0],  # Ankle Angle
            'torque': mat_data['torque'][0, 0][0][0],  # Torque
        }
    except Exception as e:
        #print(f"Error loading {file_path}: {e}")
        return None

def compare_group(files, data_directory):
    """Compare all files within a group (_left or _right) and return comparison results."""
    results = []
    file_paths = [os.path.join(data_directory, f) for f in files]
    
    for file1, file2 in combinations(file_paths, 2):
        data1 = load_mat_file(file1)
        data2 = load_mat_file(file2)
        
        if data1 is None or data2 is None:
            continue

        comparison = {'file1': file1, 'file2': file2, 'metrics': {}}
        for key in ['vGRF', 'angle']:
            if key in data1 and key in data2:
                mse, correlation = compare_signals(data1[key], data2[key])
                comparison['metrics'][key] = {'mse': mse, 'correlation': correlation}
        
        results.append(comparison)
    
    return results

def process_all_subjects(data_directory):
    """Compare all _left and _right files in the directory."""
    files = os.listdir(data_directory)
    left_files = sorted([f for f in files if '_left.mat' in f])
    right_files = sorted([f for f in files if '_right.mat' in f])

    print("Comparing all _left.mat files...")
    left_results = compare_group(left_files, data_directory)

    print("Comparing all _right.mat files...")
    right_results = compare_group(right_files, data_directory)

    return left_results, right_results

# Example usage
if __name__ == "__main__":
    data_dir = "C:\\Users\\Grégoire\\OneDrive\\Bureau\\EPF\\BRL\\Machine learning\\GaitPhaseEstimatorData\\dataHealthy[1]"
    output_dir = "C:\\Users\\Grégoire\\OneDrive\\Bureau\\EPF\\BRL\\Machine learning\\GaitPhaseEstimatorData\\dataHealthy[1]\\MergedData"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process and merge compatible files
    left_results, right_results = process_all_subjects(data_dir)
    merged_left_files = merge_files_if_compatible(left_results, output_dir)
    merged_right_files = merge_files_if_compatible(right_results, output_dir)

    print("Merged files:")
    print("Left group:", merged_left_files)
    print("Right group:", merged_right_files)
