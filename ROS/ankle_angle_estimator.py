#!/usr/bin/env python3

import rospy
import numpy as np
from ankle_exoskeleton.msg import IMUData
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation as R
import math

class AnkleAngleEstimator:
    def __init__(self):
        rospy.init_node('ankle_angle_estimator', anonymous=True)

        self.foot_imu_sub = rospy.Subscriber('/foot/imu_data', IMUData, self.foot_imu_callback)
        self.shank_imu_sub = rospy.Subscriber('/shank/imu_data', IMUData, self.shank_imu_callback)

        self.ankle_angle_pub = rospy.Publisher('/ankle_joint/angle', Float64, queue_size=2)

        self.foot_imu_data = None
        self.shank_imu_data = None

        self.initial_pos = {
            'ankle_angles': []
        }

        self.initialized = False
        self.initialization_duration = 5  # seconds
        self.start_time = rospy.Time.now()

        self.initial_angle = 0

    def foot_imu_callback(self, msg):
        self.foot_imu_data = msg
        self.process_data()

    def shank_imu_callback(self, msg):
        self.shank_imu_data = msg
        self.process_data()

    def process_data(self):
        if self.foot_imu_data and self.shank_imu_data:
            q_foot = R.from_quat([
                self.foot_imu_data.quat_x,
                self.foot_imu_data.quat_y,
                self.foot_imu_data.quat_z,
                self.foot_imu_data.quat_w
            ])

            q_shank = R.from_quat([
                self.shank_imu_data.quat_x,
                self.shank_imu_data.quat_y,
                self.shank_imu_data.quat_z,
                self.shank_imu_data.quat_w
            ])

            if not self.initialized:
                rospy.loginfo("Estimating Initial Posture")
                self.initialize(q_foot, q_shank)
            else:
                ankle_angle = self.calculate_ankle_angle(q_foot, q_shank) - self.initial_angle
                self.publish_ankle_angle(ankle_angle)

    def initialize(self, q_foot, q_shank):
        if rospy.Time.now() - self.start_time < rospy.Duration(self.initialization_duration):
            ankle_angle = self.calculate_ankle_angle(q_foot, q_shank)
            self.initial_pos['ankle_angles'].append(ankle_angle)
        else:
            ankle_angles = np.array(self.initial_pos['ankle_angles'])
            ankle_angles_filt = self.smooth_data(ankle_angles)
            self.initial_angle = np.median(ankle_angles_filt) #+ 0.16  # Compensator
            self.initialized = True
            rospy.loginfo("Publishing Ankle Angle")

    def calculate_ankle_angle(self, q_foot, q_shank):
        # Calculate distance (angle difference) between two quaternions
        q_relative = q_foot.inv() * q_shank
        return q_relative.magnitude()

    def smooth_data(self, data):
        window_size = int(0.1 * len(data))
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def publish_ankle_angle(self, ankle_angle):
        msg = Float64()
        msg.data = math.degrees(ankle_angle)
        self.ankle_angle_pub.publish(msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        estimator = AnkleAngleEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
