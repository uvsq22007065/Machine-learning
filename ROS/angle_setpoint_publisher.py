#!/usr/bin/env python3

import rospy
import math
from std_msgs.msg import Float64, Int8

class AnkleSetPointPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('ankle_joint_controller', anonymous=True)

        # Parameters
        self.use_external_reference = rospy.get_param('~use_external_reference', False)
        self.threshold = 2.0   #Degrees

        # Initialize variables
        self.current_angle = None
        self.external_reference = None
        self.time_start = rospy.get_time()

        # Publishers
        self.theta_error_pub = rospy.Publisher('/ankle_joint/theta_error', Float64, queue_size=2)
        self.goal_angle_pub = rospy.Publisher('/ankle_joint/goal_angle', Float64, queue_size=2)
        self.switching_command_pub = rospy.Publisher('/switching_command', Int8, queue_size=2)

        # Subscribers
        rospy.Subscriber('/ankle_joint/angle', Float64, self.angle_callback)
        if self.use_external_reference:
            rospy.Subscriber('/reference', Float64, self.external_reference_callback)

        # ROS rate
        self.rate = rospy.Rate(100)  # 100 Hz

    def angle_callback(self, msg):
        self.current_angle = msg.data

    def external_reference_callback(self, msg):
        self.external_reference = msg.data

    def calculate_sinusoidal_reference(self):
        amplitude = 10  # Degrees
        frequency = 0.1   # Hz    
        current_time = rospy.get_time() - self.time_start
        # Sinusoidal reference 
        reference_angle = amplitude * math.sin(2 * math.pi * frequency * current_time)
        return reference_angle

    def compute_error(self):
        if self.use_external_reference:
            reference_angle = self.external_reference
        else:
            reference_angle = self.calculate_sinusoidal_reference()

        if (self.current_angle != None):
            # Calculate error
            error = reference_angle - self.current_angle

            # Publish theta error and goal angle
            self.theta_error_pub.publish(error)
            self.goal_angle_pub.publish(reference_angle)

            # Determine switching command
            if abs(error) <= self.threshold:
                command = 0  # Zero error
            elif error > 0:
                command = 1  # Positive error
            else:
                command = 2  # Negative error

            # Publish switching command
            self.switching_command_pub.publish(command)       
            self.current_angle = None  

    def run(self):
        rospy.loginfo("Starting Angle Error Calculator")
        while not rospy.is_shutdown():
            self.compute_error()
            self.rate.sleep()
        rospy.loginfo("Angle Error Calculator Finished")

if __name__ == '__main__':
    try:
        controller = AnkleSetPointPublisher()
        controller.run()
    except rospy.ROSInterruptException:
        pass
