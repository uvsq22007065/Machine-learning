#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
import signal
import sys

def publish_desired_values():
    rospy.init_node('desired_values_publisher')
    rate = rospy.Rate(50)

    desired_values = [-0.79, -0.79, -2.78, -5.16, -6.75, -6.35, -5.76, -4.37, -3.77, -2.38,
                      -1.06, 0.33, 1.92, 3.31, 4.9, 6.69, 8.47, 10.46, 12.64, 14.03, 15.22,
                      17.21, 19.0, 20.59, 21.98, 23.56, 24.56, 25.95, 27.14, 28.73, 30.32,
                      32.1, 33.3, 35.02, 36.21, 37.4, 38.79, 40.78, 42.17, 44.42, 46.4,
                      47.99, 50.97, 53.35, 56.73, 59.31, 61.69, 64.47, 66.26, 68.84, 70.83,
                      72.81, 74.8, 76.59, 79.17, 81.35, 83.54, 85.12, 87.51, 88.7, 90.29,
                      92.08, 93.66, 94.26, 94.66, 93.86, 92.08, 89.69, 86.91, 85.12, 82.74,
                      80.69, 77.91, 75.53, 73.14, 70.43, 68.25, 65.86, 63.88, 61.69, 58.71,
                      56.13, 52.95, 49.84, 46.86, 43.75, 41.77, 38.59, 36.01, 33.82, 31.64,
                      29.65, 27.67, 25.48, 23.3, 21.31, 19.13, 16.95, 14.76, 11.98, 8.8,
                      6.62, 4.7, 2.52, -0.07, -2.45, -4.43, -6.22, -6.02, -4.43, -3.84,
                      -3.24, -2.05, -1.65, -1.46, -1.85, -1.85, -2.45, -2.65, -2.65, -2.25,
                      -1.85, -1.06, -0.26, 0.53, 0.73, 0.93, 0.73, 0.13]

    # Normalize the desired values to have a maximum value of 5
    max_value = max(abs(val) for val in desired_values)
    normalized_values = [5 * val / max_value for val in desired_values]

    pub = rospy.Publisher('/ankle_joint/desired_torque', Float32, queue_size=10)

    def signal_handler(sig, frame):
        rospy.loginfo("Ctrl+C detected. Stopping the node...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while not rospy.is_shutdown():
        for value in normalized_values:
            pub.publish(value)
            rate.sleep()

if __name__ == '__main__':
    try:
        publish_desired_values()
    except rospy.ROSInterruptException:
        pass
