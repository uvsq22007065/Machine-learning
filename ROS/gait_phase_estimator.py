#!/usr/bin/env python3

import rospy, rospkg, rosbag
from std_msgs.msg import Float64
import os
import numpy as np
import csv

class GaitPhaseEstimator:
    def __init__(self):
        # ROS setup
        rospy.init_node('gait_phase_estimator', anonymous=True)

        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force_sub = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)

        # Paths to models and training data
        self.patient = rospy.get_param("patient", "test")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ankle_exoskeleton')

        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_model.csv")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"{self.patient}.bag")

        # Variables to track state
        self.modelLoaded = False
        self.ankle_angle = None
        self.ground_force = None
        self.learning_model = []
        # Variables à initialiser dans le __init__ :
        self.in_stance_phase = False  # Suivi de la phase actuelle
        self.phase_start_time = None   # Heure de début de la phase actuelle

    def ankle_angle_callback(self, msg):
        self.ankle_angle = msg.data

    def ground_force_callback(self, msg):
        self.ground_force = msg.data

    def train_model(self):
        rospy.loginfo(f"Training model for patient {self.patient}...")

        # Check if the bag file exists
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"No .bag file found for patient {self.patient} in {self.bag_path}. Training cannot proceed.")
            rospy.signal_shutdown("Missing .bag file for training.")
            return

        # Read bag file and extract data
        bag = rosbag.Bag(self.bag_path)
        angle_data = []
        vgrf_data = []

        for topic, msg, t in bag.read_messages(topics=['/ankle_joint/angle', '/vGRF']):
            if topic == '/ankle_joint/angle':
                angle_data.append((t.to_sec(), msg.data))
            elif topic == '/vGRF':
                vgrf_data.append((t.to_sec(), msg.data))

        bag.close()

        # Convert to numpy arrays
        angle_data = np.array(angle_data)
        vgrf_data = np.array(vgrf_data)

        # Interpolate angle data to match vGRF timestamps
        interpolated_angles = np.interp(vgrf_data[:, 0], angle_data[:, 0], angle_data[:, 1])
        interpolated_forces = vgrf_data[:, 1]

        # Initialize arrays to store derivatives and phases
        force_derivatives = np.gradient(interpolated_forces)
        angle_derivatives = np.gradient(interpolated_angles)
        gait_phases = []
        gait_progress = []

        stance_start_time = None
        swing_end_time = None
        
        # --- Calcul de la phase et de la progression dans train_model ---
        for i, force in enumerate(interpolated_forces):
            current_time = vgrf_data[i, 0]

            # Détection de la phase actuelle
            if force > 0:  # Stance Phase (prioritaire)
                phase = "Stance Phase"
                if not self.in_stance_phase:  # Transition Swing -> Stance
                    self.in_stance_phase = True
                    self.phase_start_time = current_time  # Début de la phase Stance

            else:  # Swing Phase (tout le reste)
                phase = "Swing Phase"
                if self.in_stance_phase:  # Transition Stance -> Swing
                    self.in_stance_phase = False
                    self.phase_start_time = current_time  # Début de la phase Swing

            # Calcul de la progression linéaire au sein de la phase actuelle
            if self.phase_start_time is not None:
                time_in_phase = current_time - self.phase_start_time

                # Durée estimée d'une phase complète (ajustable si nécessaire)
                estimated_phase_duration = 1.0  # Durée approximative en secondes

                # Calcul de la progression linéaire (0 à 100 %)
                progress = min((time_in_phase / estimated_phase_duration) * 100, 100)
            else:
                progress = 0

            gait_phases.append(phase)
            gait_progress.append(progress)

        # --- Write data to CSV ---
        rospy.loginfo(f"Saving gait data to {self.model_path}...")
        with open(self.model_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Force', 'Force_Derivative', 'Angle', 'Angle_Derivative', 'Gait_Progress', 'Phase'])

            for i in range(len(vgrf_data)):
                writer.writerow([
                    vgrf_data[i, 0],  # Time
                    interpolated_forces[i],  # Force
                    force_derivatives[i],  # Force derivative
                    interpolated_angles[i],  # Angle
                    angle_derivatives[i],  # Angle derivative
                    gait_progress[i],  # Gait progress
                    gait_phases[i]  # Phase
                ])

        rospy.loginfo(f"Gait data saved successfully.")

        # Training code here
        
        
        # Final of Training code
        # Load the learning model
        with open(self.model_path, 'r') as model_file:
            for line in model_file:
                self.learning_model.append(line.strip().split(','))  # Assuming CSV with commas
        self.modelLoaded = True

    def estimate_phase(self):
        if self.modelLoaded and self.ankle_angle is not None and self.ground_force is not None:
            rospy.loginfo(f"Estimating phase for patient {self.patient} using model {self.model_path}...")
            # Estimation code goes here

            # Final Estimation code
            self.ankle_angle = None
            self.ground_force = None

    def run(self):
        if os.path.exists(self.model_path):
            rospy.loginfo(f"Model found for patient {self.patient}. Proceeding with phase estimation.")
            with open(self.model_path, 'r') as model_file:
                for line in model_file:
                    self.learning_model.append(line.strip().split(','))
            self.modelLoaded = True
        else:
            rospy.logwarn(f"Model not found for patient {self.patient}. Training a new model.")
            self.train_model()

        rospy.spin()


if __name__ == '__main__':
    try:
        estimator = GaitPhaseEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
