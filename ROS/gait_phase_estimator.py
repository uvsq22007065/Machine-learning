#!/usr/bin/env python3

import rospy, rospkg, rosbag
from std_msgs.msg import Float64, String
import os
import numpy as np

class GaitPhaseEstimator:
    def __init__(self):

        # ROS variables
        rospy.init_node('gait_phase_estimator', anonymous=True)

        self.ankle_angle_sub = rospy.Subscriber('/ankle_joint/angle', Float64, self.ankle_angle_callback)
        self.ground_force = rospy.Subscriber('/vGRF', Float64, self.ground_force_callback)

        # Publisher
        self.phase_pub = rospy.Publisher('/gait_phase_detection', String, queue_size=10)

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

        # Estimate gait phases from the vGRF
        def estimate_gait_phases(self, interpolated_forces):
            heel_force = interpolated_forces[:, 0:5].sum(axis=1)
            mid_force = interpolated_forces[:, 5:11].sum(axis=1)
            toe_force = interpolated_forces[:, 11:16].sum(axis=1)

            true_heel_force = np.zeros_like(heel_force)
            true_mid_force = np.zeros_like(mid_force)
            true_toe_force = np.zeros_like(toe_force)
            total_force = heel_force + mid_force + toe_force
            known_force = 1  # Known calibrated force

            gait_phases = []
            in_cycle_TO = False  # Indicates if we are in a cycle
            in_cycle_FF = False

            # Iterate through the forces to estimate the gait phases
            for i in range(len(heel_force)):
                if total_force[i] > 333:  # Apply force threshold
                    true_heel_force[i] = known_force * heel_force[i] / total_force[i]
                    true_mid_force[i] = known_force * mid_force[i] / total_force[i]
                    true_toe_force[i] = known_force * toe_force[i] / total_force[i]

                    if true_heel_force[i] > 0.2 and (true_mid_force[i] < 0.1 or true_toe_force[i] < 0.1):
                        phase = 'HS'  # Heel Strike
                        if not in_cycle_TO and not in_cycle_FF:
                            in_cycle_TO = True
                            in_cycle_FF = True

                    elif true_heel_force[i] < 0.1 and true_mid_force[i] < 0.1 and true_toe_force[i] < 0.1:
                        phase = 'MSW'  # Mid-Swing

                    elif true_heel_force[i] < 0.4 and true_mid_force[i] < 0.3 and true_toe_force[i] < 0.5:
                        if in_cycle_TO:
                            phase = 'TO'  # Toe-Off
                            in_cycle_TO = False

                    elif true_mid_force[i] > 0.3 and true_heel_force[i] < 0.3 and true_toe_force[i] > 0.25:
                        phase = 'HO'  # Heel-Off

                    elif true_heel_force[i] > 0.25 and true_mid_force[i] > 0.25:
                        if in_cycle_FF:
                            phase = 'FF/MST'  # Flat Foot/Mid-Stance
                            in_cycle_FF = False
                else:
                    phase = 'None'

                gait_phases.append(phase)

            return gait_phases


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
            gait_phases = self.estimate_gait_phases(self.ground_force)
            for i, phase in enumerate(gait_phases):
                rospy.loginfo(f"Time Step {i}: Gait Phase - {phase}")
                self.phase_pub.publish(phase)
            else:
                rospy.logwarn("Missing data")



            #Final Estimation code
            ankle_angle = None
            ground_force = None

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
