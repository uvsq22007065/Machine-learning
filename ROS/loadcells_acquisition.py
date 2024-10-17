#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32, Float32MultiArray
import serial, os

class AnkleExoskeletonNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('loadcells_acquisition_node', anonymous=True)

        # Publisher for tendon force and joint torque
        self.tendons_force = rospy.Publisher('ankle_exo/tendons_force', Float32MultiArray, queue_size=2)
        self.joint_torque_pub = rospy.Publisher('ankle_joint/net_torque', Float32, queue_size=2)

        # Serial port configuration
        self.serial_port = "/dev/ttyACM0"
        self.baud_rate = 250000
        os.system("sudo chmod 777 " + self.serial_port)
        self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)

        # Distances in meters
        self.frontal_distance = 0.19  # 19 cm
        self.posterior_distance = 0.15  # 15 cm

        # ROS rate
        self.rate = rospy.Rate(100)  # 100 Hz

    def calculate_torque(self, force, distance):
        return force * distance

    def run(self):
        rospy.loginfo("Publishing Tendons Force and Joint Torque Data")
        while not rospy.is_shutdown():
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    forces = line.split(',')
                    if len(forces) == 2:
                        frontal_force = round(float(forces[0]))
                        posterior_force = round(float(forces[1]))

                        if frontal_force < 0:
                            frontal_force = 0
                        if posterior_force < 0:
                            posterior_force = 0

                        # Calculate torques
                        frontal_torque = self.calculate_torque(frontal_force, self.frontal_distance)
                        posterior_torque = self.calculate_torque(posterior_force, self.posterior_distance)

                        # Publish tendon forces
                        msg = Float32MultiArray()
                        msg.data = [frontal_force, posterior_force]
                        self.tendons_force.publish(msg)

                        # Publish joint torque
                        self.joint_torque_pub.publish(frontal_torque-posterior_torque)

            except serial.SerialException as e:
                rospy.logerr(f"Serial port error: {e}")
            except ValueError as e:
                rospy.logerr(f"Value error: {e}")

            self.rate.sleep()
        rospy.loginfo("Force and Torque Data Finished")

if __name__ == '__main__':
    try:
        node = AnkleExoskeletonNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
