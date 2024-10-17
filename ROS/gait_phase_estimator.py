#!/usr/bin/env python3

import rospy, rospkg, rosbag
from std_msgs.msg import Float64
import os
import numpy as np

class GaitPhaseEstimator:
    def __init__(self):

        # ROS variables
        rospy.init_node('gait_phase_estimator', anonymous=True)

        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)

        # Learning model path
        self.patient = rospy.get_param("patient","test")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ankle_exoskeleton') 
        self.model_path = os.path.join(package_path, "log", "learning_models", f"{self.patient}_model.csv")
        self.bag_path = os.path.join(package_path, "log", "training_bags", f"{self.patient}.bag")

        self.modelLoaded = False

        self.ankle_angle = []
        self.ground_force = []
        self.learning_model = []

    def ankle_angle_callback(self, msg):
        self.ankle_angle = msg.data
        self.estimate_phase()

    def ground_force_callback(self, msg):
        self.ground_force = msg.data
        self.estimate_phase()

    def train_model(self):
        rospy.loginfo(f"Training model for patient {self.patient}...")
        
        # Check if the bag file exists
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"No .bag file found for patient {self.patient} in {self.bag_path}. Training cannot proceed.")
            rospy.signal_shutdown("Missing .bag file for training.")
            return

        # Create the dataset for training
        # Read bag file
        bag = rosbag.Bag(self.bag_path)
        angle_data = []
        vgrf_data = []

        # Extract data from the /ankle_joint/angle and /vGRF topics with timestamps
        for topic, msg, t in bag.read_messages(topics=['/ankle_joint/angle', '/vGRF']):
            if topic == '/ankle_joint/angle':
                angle_data.append((t.to_sec(), msg.data))
            elif topic == '/vGRF':
                vgrf_data.append((t.to_sec(), msg.data))
        
        bag.close()

        # Convert to numpy arrays for interpolation
        angle_data = np.array(angle_data)
        vgrf_data = np.array(vgrf_data)

        # Interpolate angle data to match vGRF timestamps (angle data have lower frequency than vGRF)
        interpolated_angles = np.interp(vgrf_data[:, 0], angle_data[:, 0], angle_data[:, 1])
        interpolated_forces = vgrf_data[:, 1]
        

        # Training code goes here



        # Final of Traning code
        # Load the learning model 
        with open(self.model_path, 'r') as model_file:
            for line in model_file:
                self.learning_model.append(line.strip().split(','))  # Assuming CSV with commas
        self.modelLoaded = True
        
    def estimate_phase(self):
        if ((self.modelLoaded == True) and (self.ankle_angle != None) and (self.ground_force != None)):
            rospy.loginfo(f"Estimating phase for patient {self.patient} using model {self.model_path}...")
            # Estimation code goes here
            

            # Déterminer la phase actuelle en fonction du vGRF
            if self.ground_force <= 0:
                self.gait_phase = "Swing Phase"
                previous_phase = False
            else:
                self.gait_phase = "Stance Phase"
                previous_phase = True

            # Mise à jour du pourcentage d'accomplissement du cycle de marche
            if previous_phase:
                if self.gait_phase == "Stance Phase":
                    self.gait_progress = 0  # Début du cycle (stance phase)
                elif self.gait_phase == "Swing Phase":
                    self.gait_progress = 100  # Fin du cycle (swing phase)
            
            # Affichage du progrès par pas de 10 %
            if self.gait_progress in range(0, 101, 10):
                rospy.loginfo(f"Gait Progress: {self.gait_progress}%")

            # Incrément du progrès dans la phase actuelle
            if self.gait_phase == "Stance Phase" and self.gait_progress < 50:
                self.gait_progress += 10
            elif self.gait_phase == "Swing Phase" and self.gait_progress < 100:
                self.gait_progress += 10

            rospy.loginfo(f"Current Phase: {self.gait_phase}, Progress: {self.gait_progress}%")

            # Réinitialisation des variables après estimation
        self.ankle_angle = None
        self.ground_force = None



    def run(self):
        # Check if the model exists
        if os.path.exists(self.model_path):
            rospy.loginfo(f"Model found for patient {self.patient}. Proceeding with phase estimation.")
            # Load the learning model 
            with open(self.model_path, 'r') as model_file:
                for line in model_file:
                    self.learning_model.append(line.strip().split(','))  # Assuming CSV with commas
            self.modelLoaded = True
        else:
            rospy.logwarn(f"Model not found for patient {self.patient}. Training a new model.")
            self.modelLoaded = False
            self.train_model()

        rospy.spin()


if __name__ == '__main__':
    try:
        estimator = GaitPhaseEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
