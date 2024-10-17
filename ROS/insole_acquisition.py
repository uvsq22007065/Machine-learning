#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
import os
#from serial import Serial
import serial

class ArduinoNode:
    def __init__(self):

        rospy.init_node('insole_acquisition', anonymous=True)

        # Ros Params
        usb_port = rospy.get_param('~usb_port', 'ttyUSB1')
        baudrate = rospy.get_param('~baudrate', 250000)

        os.system("sudo chmod 777 /dev/{}".format(usb_port))
        #os.system("sudo chmod 777 -R /sys/bus/usb-serial/devices/{}/latency_timer".format(usb_port))
        #os.system("echo 1 > /sys/bus/usb-serial/devices/{}/latency_timer".format(usb_port))

        self.ser = serial.Serial("/dev/" + usb_port, baudrate, timeout=1)

        self.adc_pub = rospy.Publisher('insole_data', Int32MultiArray, queue_size=10)

        self.rate = rospy.Rate(500)

    def read_data_from_arduino(self):
        incomplete_data_count = 0
        while not rospy.is_shutdown():
            data = self.ser.readline().decode().strip().split('\t')
            #print((data))
            if len(data) == 16:
                p15, p4, p14, p13, p11, p10, p16, p7, p8, p12, p5, p3, p2, p6, p1, p9 = map(int, data)

                msg = Int32MultiArray(data=[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16])
                self.adc_pub.publish(msg)

            else:
                incomplete_data_count += 1
                if incomplete_data_count >= 5:
                    rospy.logwarn("Closing and reopening port due to incomplete data")
                    self.ser.close()
                    self.ser.open()
                    incomplete_data_count = 0

            self.rate.sleep()



    def run(self):
        try:
            rospy.loginfo('Starting Insole Acquisition')
            self.read_data_from_arduino()
        finally:
            rospy.loginfo('Closing Serial Port')
            self.ser.close()

if __name__ == '__main__':
    arduino_node = ArduinoNode()
    arduino_node.run()
