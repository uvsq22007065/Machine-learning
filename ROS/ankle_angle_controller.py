#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int8, Int32
import time

class AnkleAngleController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('pd_controller', anonymous=True)

        # Gains for frontal and posterior motors
        self.kp_frontal = rospy.get_param('~kp_frontal', 5.0)  # Default value for frontal motor
        self.kd_frontal = rospy.get_param('~kd_frontal', 0)  # Default value for frontal motor
        self.kp_posterior = rospy.get_param('~kp_posterior', 5.0)  # Default value for posterior motor
        self.kd_posterior = rospy.get_param('~kd_posterior', 0)  # Default value for posterior motor

        # Subscribers
        rospy.Subscriber('/ankle_joint/theta_error', Float64, self.theta_error_callback)
        rospy.Subscriber('/switching_command', Int8, self.switching_command_callback)

        # Publishers for motor goal velocities
        self.frontal_velocity_pub = rospy.Publisher('/ankle_exo/frontal/dynamixel_motor/goal_velocity_angleLoop', Int32, queue_size=2)
        self.posterior_velocity_pub = rospy.Publisher('/ankle_exo/posterior/dynamixel_motor/goal_velocity_angleLoop', Int32, queue_size=2)

        # Previous errors and time for PD control
        self.prev_theta_error = 0.0
        self.prev_time = time.time()

        # Switching command state
        self.switching_command = 0  # 0 means no control

        # RPM limits
        self.max_rpm = 54.0
        self.min_rpm = -54.0

        # Conversion factor from RPM to control value
        self.rpm_to_value = 1 / 0.229

        # ROS rate
        self.rate = rospy.Rate(100)  # 100 Hz

    def theta_error_callback(self, msg):
        self.theta_error = msg.data

    def switching_command_callback(self, msg):
        self.switching_command = msg.data

    def calculate_velocity(self, error, prev_error, kp, kd):
        current_time = time.time()
        dt = current_time - self.prev_time

        # Proportional and Derivative terms
        proportional = kp * error
        derivative = kd * (error - prev_error) / dt if dt > 0 else 0.0

        velocity = proportional + derivative

        # Clamp the velocity within the limits
        if velocity > self.max_rpm:
            velocity = self.max_rpm
        if velocity < self.min_rpm:
            velocity = self.min_rpm

        # Convert RPM to control value
        self.prev_time = current_time

        return round(velocity * self.rpm_to_value)

    def run(self):
        rospy.loginfo("PD Controller Started")

        while not rospy.is_shutdown():
            if hasattr(self, 'theta_error'):
                # Get the absolute value of the error
                abs_error = abs(self.theta_error)

                # Control logic based on switching command
                if self.switching_command == 0:  # Error is zero, no control applied
                    frontal_velocity = 0
                    posterior_velocity = 0
                elif self.switching_command == 1:  # Positive error, control only frontal motor
                    frontal_velocity = self.calculate_velocity(abs_error, self.prev_theta_error, self.kp_frontal, self.kd_frontal)
                    posterior_velocity = 0
                elif self.switching_command == 2:  # Negative error, control only posterior motor
                    frontal_velocity = 0
                    posterior_velocity = self.calculate_velocity(abs_error, self.prev_theta_error, self.kp_posterior, self.kd_posterior)

                # Publish velocities
                self.frontal_velocity_pub.publish(frontal_velocity)
                self.posterior_velocity_pub.publish(posterior_velocity)

                # Update previous values
                self.prev_theta_error = abs_error
                self.prev_time = time.time()

                self.switching_command = 0

            # Sleep to maintain loop rate
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = AnkleAngleController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
