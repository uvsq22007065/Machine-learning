#!/usr/bin/env python3

import rospy
import serial, os, time
import numpy as np
from std_msgs.msg import Float64, Bool
import statistics

class TorqueForceSensorATI:
    def __init__(self, port='/dev/ttyUSB2'):
        self.port = port
        self.ser = None

        os.system("sudo chmod 777 " + str(self.port))

        ''' Sensor Parameters '''
        self.cv_parameter = 4 #Limit variables sent by sensor (Example 04 -> Only Fz)
        self.data_length = 1 #Parameter conditioned to the cv_parameter (If it is only Fz the length is 1)
        self.force_sensor_constant = 320.0 #Review datasheet: Constant used to convert counts to force value
        self.torque_sensor_constant = 5332.8 #Review datasheet: Constant used to convert counts to torque value

        ''' Filter Parameters '''
        self.filter_size = 5

        ''' ROS Parameters'''
        rospy.init_node('torque_force_sensor_node')
        self.force_pub = rospy.Publisher('force_torque_sensor_data/fz', Float64, queue_size=1)
        self.reset_sub = rospy.Subscriber('force_torque_sensor_data/reset_sensor', Bool, self.reset_sensor)

        ''' Code Variables '''
        self.data = []
        self.filtered_data = []


    def connect(self):
        try:
            self.ser = serial.Serial(self.port, baudrate=115200, timeout=1)
            rospy.loginfo('Torque-Force Sensor connected on port %s', self.port)
            return True
        except serial.SerialException:
            self.ser.close()
            rospy.logerr('Failed to connect to Torque-Force Sensor on port %s', self.port)
            return False

    def initialization_commands(self):
        try:
            #Please review datasheet to include or change sensor functionalities
            self.ser.write(b'CB 115200\r') #change the baudrate, uncomment once time
            self.ser.write(b'CD R\r') #Data Type for the communication
            if self.cv_parameter == 4:
                self.ser.write(b'CV 04\r')  #Limit variables transmitted by sensor
            self.ser.write(b'SB\r')   #Reset force data to zero
            self.ser.write(b'QS\r')   #Activate sensor communication
            rospy.loginfo('Configuration completed successfully. Torque-Force Sensor sending data...')
            return True
        except:
            self.ser.write(b'\r')
            self.ser.close()
            rospy.logerr('Failed to configure Torque-Force Sensor')
            return False

    def reset_sensor(self, flag):
        if flag.data:
            try:
                self.ser.write(b'\r')
                self.ser.write(b'SB\r')
                rospy.loginfo('Torque-Force Sensor reset')
                time.sleep(0.1)
                self.ser.write(b'QS\r')
            except serial.SerialException:
                rospy.logerr('Failed to reset Torque-Force Sensor')


    def acquire_data(self):
        try:
            if len(self.data) < self.filter_size:
                d = self.ser.readline().strip().decode('ascii')
                values = []
                for v in d[3::].split(','):
                    try:
                        values.append(float(v))
                    except ValueError:
                        return False
                if len(values) == self.data_length:
                    self.data.append(values)
                    return
                else:
                    return
            else:
                self.filtered_data = np.mean(np.array(self.data).T, axis=1).tolist()
                #print("Filtered: " + str(self.filtered_data))
                self.data.pop(0)
                return True
            return False

        except serial.SerialException:
            rospy.logerr('Failed to acquire data from Torque-Force Sensor')
            return False

    def force_estimation(self):
        force = float(self.filtered_data[-1]/self.force_sensor_constant)
        #print((force))
        return force



if __name__ == '__main__':
    sensor = TorqueForceSensorATI()
    connected = sensor.connect()
    if connected:
        configured = sensor.initialization_commands()
        if configured:
            rate = rospy.Rate(800) # 800 Hz
            while not rospy.is_shutdown():
                acquired = sensor.acquire_data()
                if acquired:
                    #TODO: Estimate and publish for multiples force values
                    if len(sensor.filtered_data) == 1:
                        force = round(-sensor.force_estimation(),3)
                        sensor.force_pub.publish(force)
                rate.sleep()
        sensor.ser.close()
        rospy.loginfo('Torque-Force Sensor node stopped')
