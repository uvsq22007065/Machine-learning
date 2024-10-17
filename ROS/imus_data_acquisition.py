#!/usr/bin/env python3

import rospy
import serial, os
from ankle_exoskeleton.msg import IMUData

class IMUBNOArduinoNode:
    def __init__(self, port):
        rospy.init_node("imus_data_acq", anonymous=True)
        self.imuF_pub = rospy.Publisher("foot/imu_data", IMUData, queue_size=1)
        self.imuS_pub = rospy.Publisher("shank/imu_data", IMUData, queue_size=1)
        self.serial_port = None
        self.calibrated = False
        self.port = port
        os.system("sudo chmod 777 " + self.port)
        self.calibrate_imu()

    def calibrate_imu(self):
        # Open serial port to Arduino
        self.serial_port = serial.Serial(self.port, 1000000)
        rospy.loginfo("Calibrating IMUs on port {}...".format(self.port))
        while not rospy.is_shutdown():
            calibration_str = self.serial_port.readline().strip().decode()
            #print(len(calibration_str))
            if len(calibration_str) > 20:
                rospy.loginfo("IMU on port {} already calibrated!".format(self.port))
                self.calibrated = True
                break
            try:
                systemF, gyroF, accelF, magF, systemS, gyroS, accelS, magS = map(int, calibration_str.split(","))
                rospy.loginfo("Calibration on port {}: SysF=%d GyroF=%d AccelF=%d MagF=%d SysS=%d GyroS=%d AccelS=%d MagS=%d".format(self.port), systemF, gyroF, accelF, magF, systemS, gyroS, accelS, magS)
                if systemF == -28 and gyroF == -28 and accelF == -28 and magF == -28:
                    rospy.loginfo("Foot IMU on port {} not found!".format(self.port))
                if systemS == -29 and gyroS == -29 and accelS == -29 and magS == -29:
                    rospy.loginfo("Shank IMU on port {} not found!".format(self.port))
                # Check if IMU is fully calibrated
                if systemF == 3 and gyroF == 3 and accelF == 3 and magF == 3 and systemS == 3 and gyroS == 3 and accelS == 3 and magS == 3:
                    rospy.loginfo("IMUs on port {} calibrated!".format(self.port))
                    self.calibrated = True
                    break
            except:
                rospy.logerr("Incomplete Data")

    def publish_imu_data(self):
        if not self.calibrated:
            return

        try:
            data_str = self.serial_port.readline().strip().decode()
            #oxF, oyF, ozF, gxF, gyF, gzF, axF, ayF, azF, gvxF, gvyF, gvzF, qxF, qyF, qzF, qwF, oxS, oyS, ozS, gxS, gyS, gzS, axS, ayS, azS, gvxS, gvyS, gvzS, qxS, qyS, qzS, qwS = map(float, data_str.split(","))
            gxF, gyF, gzF, axF, ayF, azF, gvxF, gvyF, gvzF, qxF, qyF, qzF, qwF, gxS, gyS, gzS, axS, ayS, azS, gvxS, gvyS, gvzS, qxS, qyS, qzS, qwS = map(float, data_str.split(","))

            # Publish sensor data
            imu_msg = IMUData()
            imu_msg.accel_x = axF
            imu_msg.accel_y = ayF
            imu_msg.accel_z = azF
            imu_msg.gyro_x = gxF
            imu_msg.gyro_y = gyF
            imu_msg.gyro_z = gzF
            imu_msg.quat_x = qxF
            imu_msg.quat_y = qyF
            imu_msg.quat_z = qzF
            imu_msg.quat_w = qwF
            #imu_msg.euler_x = oxF
            #imu_msg.euler_y = oyF
            #imu_msg.euler_z = ozF
            imu_msg.gravity_x = gvxF
            imu_msg.gravity_y = gvyF
            imu_msg.gravity_z = gvzF

            self.imuF_pub.publish(imu_msg)

            imu_msg = IMUData()
            imu_msg.accel_x = axS
            imu_msg.accel_y = ayS
            imu_msg.accel_z = azS
            imu_msg.gyro_x = gxS
            imu_msg.gyro_y = gyS
            imu_msg.gyro_z = gzS
            imu_msg.quat_x = qxS
            imu_msg.quat_y = qyS
            imu_msg.quat_z = qzS
            imu_msg.quat_w = qwS
            #imu_msg.euler_x = oxS
            #imu_msg.euler_y = oyS
            #imu_msg.euler_z = ozS
            imu_msg.gravity_x = gvxS
            imu_msg.gravity_y = gvyS
            imu_msg.gravity_z = gvzS

            self.imuS_pub.publish(imu_msg)
        except ValueError as e:
            rospy.logwarn(f"Failed to parse IMU data: {e}")
        except serial.SerialException as e:
            rospy.logerr(f"Serial port error: {e}")

if __name__ == '__main__':
    try:
        port = rospy.get_param('port_name', '/dev/ttyACM1')
        imu_node = IMUBNOArduinoNode(port)
        rate = rospy.Rate(500)
        while not rospy.is_shutdown():
            imu_node.publish_imu_data()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
