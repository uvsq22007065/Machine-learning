#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray, Int32, Int8
# from ankle_exoskeleton.srv import DynamixelCmdSimplified, DynamixelCmdSimplifiedRequest
import time

class TendonForceController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('tendon_force_controller', anonymous=True)

        self.desired_tendon_force = rospy.get_param('~desired_tendon_force', 5)  # Default value is 15N

        # Publishers for motor goal velocities
        self.frontal_velocity_pub = rospy.Publisher('/ankle_exo/frontal/dynamixel_motor/goal_velocity_forceLoop', Int32, queue_size=2)
        self.posterior_velocity_pub = rospy.Publisher('/ankle_exo/posterior/dynamixel_motor/goal_velocity_forceLoop', Int32, queue_size=2)

        # Subscriber for tendons force and switching commands
        rospy.Subscriber('ankle_exo/tendons_force', Float32MultiArray, self.tendons_force_callback)
        rospy.Subscriber('switching_command', Int8, self.switching_command_callback)

        # RPM limits
        self.max_rpm = 54.0
        self.min_rpm = -54.0

        # Conversion factor from RPM to control value
        self.rpm_to_value = 1 / 0.229

        # Distances in meters
        self.frontal_distance = 0.19  # 19 cm
        self.posterior_distance = 0.15  # 15 cm

        # Current tendon forces variables
        self.frontal_force = None
        self.posterior_force = None

        # Previous errors and times for PD control
        self.prev_frontal_error = 0.0
        self.prev_posterior_error = 0.0
        self.prev_time = time.time()

        #Switch Initialization
        self.sw_cmd = 0 #Angle Error is zero

        # ROS rate
        self.rate = rospy.Rate(100)  # 100 Hz

        # Activate motors on start
        #self.activate_motors()

        # Set shutdown hook
        #rospy.on_shutdown(self.shutdown_hook)

    # def activate_motors(self):
    #     try:
    #         rospy.wait_for_service('/ankle_exo/frontal/dynamixel_motor/torque_enable')
    #         rospy.wait_for_service('/ankle_exo/posterior/dynamixel_motor/torque_enable')

    #         enable_torque_frontal = rospy.ServiceProxy('/ankle_exo/frontal/dynamixel_motor/torque_enable', DynamixelCmdSimplified)
    #         enable_torque_posterior = rospy.ServiceProxy('/ankle_exo/posterior/dynamixel_motor/torque_enable', DynamixelCmdSimplified)

    #         # Create the request to enable torque (id: 0, value: 1)
    #         request = DynamixelCmdSimplifiedRequest(id=0, value=1)

    #         # Call the services to enable the torque
    #         res_frontal = enable_torque_frontal(request)
    #         res_posterior = enable_torque_posterior(request)

    #         if res_frontal.comm_result and res_posterior.comm_result:
    #             rospy.loginfo("Torque enabled for both motors.")
    #         else:
    #             rospy.logwarn("Failed to enable torque for one or both motors.")

    #     except rospy.ServiceException as e:
    #         rospy.logerr(f"Service call failed: {e}")

    def tendons_force_callback(self, msg):
        self.frontal_force = msg.data[0]
        self.posterior_force = msg.data[1]
    
    def switching_command_callback(self, msg):
        self.sw_cmd = msg.data

    def calculate_velocity(self, current_force, prev_error, desired_force, id_motor):
        # Get the current time and compute the time difference
        current_time = time.time()
        dt = current_time - self.prev_time

        # Error and threshold 
        error = round(desired_force) - round(current_force)
        threshold = 1

        # Control Gains (Relationship between lever arms)
        if id_motor == 1: #Frontal Motor
            Kp_rolling = 3.8 
            Kp_unrolling = 2
        else:             #Posterior Motor
            Kp_rolling = 2.8
            Kp_unrolling = 1

        # PD controller 
        if abs(error) > threshold:
            derivative = (error - prev_error) / dt if dt > 0 else 0.0
            if error > threshold: #Rolling Tendon
                velocity_p = Kp_rolling * error 
                velocity_d = 0.00005 * derivative
            else: #Unrolling Tendon
                velocity_p = Kp_unrolling * error
                velocity_d = 0.00005 * derivative
        else:
            error = 0
            velocity_p = 0
            velocity_d = 0
            
        velocity = velocity_p + velocity_d

        # Clamp the velocity within the limits
        if velocity > self.max_rpm:
            velocity = self.max_rpm
        if velocity < self.min_rpm:
            velocity = self.min_rpm

        # Convert RPM to control value
        self.prev_time = current_time
        return round(velocity * self.rpm_to_value), error

    def run(self):
        rospy.loginfo("Tendon Force Controller Started")
        rospy.loginfo("Desired Tendon Force: " + str(self.desired_tendon_force) + "N")
        
        while not rospy.is_shutdown():
            if ((self.frontal_force != None) and (self.posterior_force != None)):
                # Tendon Force Calculator
                self.desired_frontal_force = self.desired_tendon_force  # Newtons larger distance
                self.desired_posterior_force = self.desired_frontal_force*self.frontal_distance/self.posterior_distance # Newtons

                #Switch: Saturation of Tendon Force from the Angle Error (3 cases)
                if self.sw_cmd == 1:    #Angle error is positive 
                    if self.frontal_force > self.desired_frontal_force:
                        self.frontal_force = self.desired_frontal_force
                elif self.sw_cmd == 2:  #Angle error is negative
                    if self.posterior_force > self.desired_posterior_force:
                        self.posterior_force = self.desired_posterior_force  
                else:                   #Angle error is zero
                    pass

                frontal_velocity, self.prev_frontal_error = self.calculate_velocity(self.frontal_force, self.prev_frontal_error, self.desired_frontal_force,1)
                posterior_velocity, self.prev_posterior_error = self.calculate_velocity(self.posterior_force, self.prev_posterior_error, self.desired_posterior_force,2)

                self.frontal_velocity_pub.publish(frontal_velocity)
                self.posterior_velocity_pub.publish(posterior_velocity)

                self.frontal_force = None
                self.posterior_force = None
                self.sw_cmd = None

            self.rate.sleep()
            
        rospy.loginfo("Tendon Force Controller Finished")

    # def shutdown_hook(self):
    #     # Stop the motors by setting the velocity to 0
    #     self.frontal_velocity_pub.publish(0)
    #     self.posterior_velocity_pub.publish(0)

    #     # Call services to disable torque for both motors
    #     try:
    #         rospy.wait_for_service('/ankle_exo/frontal/dynamixel_motor/torque_enable')
    #         rospy.wait_for_service('/ankle_exo/posterior/dynamixel_motor/torque_enable')

    #         disable_torque_frontal = rospy.ServiceProxy('/ankle_exo/frontal/dynamixel_motor/torque_enable', DynamixelCmdSimplified)
    #         disable_torque_posterior = rospy.ServiceProxy('/ankle_exo/posterior/dynamixel_motor/torque_enable', DynamixelCmdSimplified)

    #         # Create the request to disable torque (id: 0, value: 0)
    #         request = DynamixelCmdSimplifiedRequest(id=0, value=0)

    #         # Call the services to disable the torque
    #         res_frontal = disable_torque_frontal(request)
    #         res_posterior = disable_torque_posterior(request)

    #         if res_frontal.comm_result and res_posterior.comm_result:
    #             rospy.loginfo("Torque disabled for both motors.")
    #         else:
    #             rospy.logwarn("Failed to disable torque for one or both motors.")

    #     except rospy.ServiceException as e:
    #         rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    try:
        controller = TendonForceController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
