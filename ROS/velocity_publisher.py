#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32
from ankle_exoskeleton.srv import DynamixelCmdSimplified, DynamixelCmdSimplifiedRequest

class VelocityPublisher:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('velocity_publisher', anonymous=True)

        # Publishers for frontal and posterior goal velocities
        self.frontal_velocity_pub = rospy.Publisher('/ankle_exo/frontal/dynamixel_motor/goal_velocity', Int32, queue_size=2)
        self.posterior_velocity_pub = rospy.Publisher('/ankle_exo/posterior/dynamixel_motor/goal_velocity', Int32, queue_size=2)

        # Variables to store values from both loops
        self.frontal_force_loop = 0
        self.frontal_angle_loop = 0
        self.posterior_force_loop = 0
        self.posterior_angle_loop = 0

        # Maximum and minimum velocity limits
        self.max_velocity = 240
        self.min_velocity = -240

        # Subscribers for the force and angle loop velocities
        rospy.Subscriber('/ankle_exo/frontal/dynamixel_motor/goal_velocity_forceLoop', Int32, self.frontal_force_loop_callback)
        rospy.Subscriber('/ankle_exo/frontal/dynamixel_motor/goal_velocity_angleLoop', Int32, self.frontal_angle_loop_callback)
        rospy.Subscriber('/ankle_exo/posterior/dynamixel_motor/goal_velocity_forceLoop', Int32, self.posterior_force_loop_callback)
        rospy.Subscriber('/ankle_exo/posterior/dynamixel_motor/goal_velocity_angleLoop', Int32, self.posterior_angle_loop_callback)

        # ROS rate for control loop
        self.rate = rospy.Rate(100)  # 100 Hz

        # Activate motors on start
        self.activate_motors()

        # Set shutdown hook
        rospy.on_shutdown(self.shutdown_hook)
    
    def activate_motors(self):
        try:
            rospy.wait_for_service('/ankle_exo/frontal/dynamixel_motor/torque_enable')
            rospy.wait_for_service('/ankle_exo/posterior/dynamixel_motor/torque_enable')

            enable_torque_frontal = rospy.ServiceProxy('/ankle_exo/frontal/dynamixel_motor/torque_enable', DynamixelCmdSimplified)
            enable_torque_posterior = rospy.ServiceProxy('/ankle_exo/posterior/dynamixel_motor/torque_enable', DynamixelCmdSimplified)

            # Create the request to enable torque (id: 0, value: 1)
            request = DynamixelCmdSimplifiedRequest(id=0, value=1)

            # Call the services to enable the torque
            res_frontal = enable_torque_frontal(request)
            res_posterior = enable_torque_posterior(request)

            if res_frontal.comm_result and res_posterior.comm_result:
                rospy.loginfo("Torque enabled for both motors.")
            else:
                rospy.logwarn("Failed to enable torque for one or both motors.")

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def frontal_force_loop_callback(self, msg):
        self.frontal_force_loop = msg.data

    def frontal_angle_loop_callback(self, msg):
        self.frontal_angle_loop = msg.data

    def posterior_force_loop_callback(self, msg):
        self.posterior_force_loop = msg.data

    def posterior_angle_loop_callback(self, msg):
        self.posterior_angle_loop = msg.data

    def clamp_velocity(self, velocity):
        # Clamp the velocity within the defined limits
        return max(self.min_velocity, min(self.max_velocity, velocity))

    def run(self):
        while not rospy.is_shutdown():
            # Calculate the sum of force and angle loops for both frontal and posterior motors
            frontal_velocity = self.frontal_force_loop + self.frontal_angle_loop
            posterior_velocity = self.posterior_force_loop + self.posterior_angle_loop

            # Clamp the velocities to ensure they are within the allowed range
            frontal_velocity = self.clamp_velocity(frontal_velocity)
            posterior_velocity = self.clamp_velocity(posterior_velocity)

            # Publish the clamped velocities
            self.frontal_velocity_pub.publish(frontal_velocity)
            self.posterior_velocity_pub.publish(posterior_velocity)

            # Sleep to maintain the loop rate
            self.rate.sleep()

    def shutdown_hook(self):
        # Stop the motors by setting the velocity to 0
        self.frontal_velocity_pub.publish(0)
        self.posterior_velocity_pub.publish(0)

        # Call services to disable torque for both motors
        try:
            rospy.wait_for_service('/ankle_exo/frontal/dynamixel_motor/torque_enable')
            rospy.wait_for_service('/ankle_exo/posterior/dynamixel_motor/torque_enable')

            disable_torque_frontal = rospy.ServiceProxy('/ankle_exo/frontal/dynamixel_motor/torque_enable', DynamixelCmdSimplified)
            disable_torque_posterior = rospy.ServiceProxy('/ankle_exo/posterior/dynamixel_motor/torque_enable', DynamixelCmdSimplified)

            # Create the request to disable torque (id: 0, value: 0)
            request = DynamixelCmdSimplifiedRequest(id=0, value=0)

            # Call the services to disable the torque
            res_frontal = disable_torque_frontal(request)
            res_posterior = disable_torque_posterior(request)

            if res_frontal.comm_result and res_posterior.comm_result:
                rospy.loginfo("Torque disabled for both motors.")
            else:
                rospy.logwarn("Failed to disable torque for one or both motors.")

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    try:
        velocity_publisher = VelocityPublisher()
        velocity_publisher.run()
    except rospy.ROSInterruptException:
        pass
