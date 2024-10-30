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
                
        # Initialisation des durées de phase
        self.estimated_stance_duration = 0.6 * 14.3 / 10  # Durée estimée de la phase Stance
        self.estimated_swing_duration = 0.4 * 14.3 / 10   # Durée estimée de la phase Swing

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
        current_time = []
        stance_time_in_phase = []
        swing_time_in_phase = []
        estimated_stance_duration = 0.6 * 14.3 / 10
        estimated_swing_duration = 0.4 * 14.3 / 10
        stance_start_time = 12
        swing_start_time = None
        previous_phase = None             # Suivi de la phase précédente
        previous_phase_time = None        # Temps de la transition entre les phases
        estimated_phase_duration = None   # Durée totale estimée d'une phase complète (Stance + Swing)
        in_stance_phase = False  # Suivi de la phase actuelle
        phase_start_time = 0   # Heure de début de la phase actuelle
        previous_phase = "Stance Phase"
        
        # Variables pour suivre la durée des phases
        total_stance_time = 0
        total_swing_time = 0
        stance_count = 0
        swing_count = 0
        stance_duration = 0.6815521717071533
        time_before_training = vgrf_data[0,0]

        for i, force in enumerate(interpolated_forces):
            current_time_value = vgrf_data[i, 0] - time_before_training
            current_time.append(current_time_value)

            # Stance Phase
            if force > 0:
                phase = "Stance Phase"
                if previous_phase == "Swing Phase":
                    # Swing duration calculations
                    if swing_count == 0:
                        estimated_swing_duration = swing_duration
                    elif swing_count > 0:
                        swing_duration = current_time[i] - swing_start_time
                        total_swing_time += swing_duration
                        estimated_swing_duration = total_swing_time / swing_count
                    # Transition: Swing -> Stance
                    in_stance_phase = True
                    previous_phase = "Stance Phase"
                    swing_time_in_phase_value = 0
                    swing_time_in_phase.append(swing_time_in_phase_value)
                    stance_start_time = current_time[i]
                    stance_time_in_phase_value = current_time[i] - stance_start_time
                    stance_time_in_phase.append(stance_time_in_phase_value)
                    stance_count += 1
                    progress = min((60 * (stance_time_in_phase[i] / estimated_stance_duration)), 60)
                else:
                    # Regular Stance Phase updates
                    swing_time_in_phase_value = 0
                    swing_time_in_phase.append(swing_time_in_phase_value)
                    stance_time_in_phase_value = current_time[i] - stance_start_time
                    stance_time_in_phase.append(stance_time_in_phase_value)
                    progress = min((60 * (stance_time_in_phase[i] / estimated_stance_duration)), 60)

            # Swing Phase
            else:
                phase = "Swing Phase"
                if previous_phase == "Stance Phase":
                    # stance duration calculations
                    if stance_count == 0:
                        estimated_stance_duration = stance_duration
                    elif stance_count == 1:
                        estimated_stance_duration = 0
                        stance_duration = current_time[i] - stance_start_time
                        total_stance_time += stance_duration
                        estimated_stance_duration = total_stance_time / stance_count
                    elif stance_count > 1:
                        stance_duration = current_time[i] - stance_start_time
                        total_stance_time += stance_duration
                        estimated_stance_duration = total_stance_time / stance_count
                    # Transition: Stance -> Swing
                    progress = min(60 + (40 * (swing_time_in_phase_value / estimated_swing_duration)), 100)
                    in_stance_phase = False
                    previous_phase = "Swing Phase"
                    stance_time_in_phase_value = 0
                    stance_time_in_phase.append(stance_time_in_phase_value)
                    swing_start_time = current_time[i]  # Update at swing phase start
                    swing_time_in_phase_value = 0
                    swing_time_in_phase.append(swing_time_in_phase_value)
                    swing_count += 1
                else:
                    # Continuously update within Swing Phase
                    stance_time_in_phase_value = 0
                    stance_time_in_phase.append(stance_time_in_phase_value)
                    swing_time_in_phase_value = current_time[i] - swing_start_time
                    swing_time_in_phase.append(swing_time_in_phase_value)
                    # Calculate progress from 60% to 100% based on swing phase duration
                    progress = min(60 + (40 * (swing_time_in_phase_value / estimated_swing_duration)), 100)

            # Append values
            gait_phases.append(phase)
            gait_progress.append(progress)

        # --- Write data to CSV ---
        rospy.loginfo(f"Saving gait data to {self.model_path}...")
        with open(self.model_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time', 'Force', 'Force_Derivative', 'Angle', 'Angle_Derivative', 'Gait_Progress', 'Phase'])

            for i in range(len(vgrf_data)):
                writer.writerow([
                    current_time[i],  # Time
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
