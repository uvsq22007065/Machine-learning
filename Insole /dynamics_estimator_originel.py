#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from custom_msgs.msg import InsoleData  # Assuming you have custom message types for insole data

class GaitPhaseDetectionNode:
    def __init__(self):
        rospy.init_node('gait_phase_detection_node', anonymous=True)

        # Subscriber to the insole data stream
        self.sub = rospy.Subscriber("/insole_data", InsoleData, self.callback)

        # Publisher for detected gait phases
        self.pub = rospy.Publisher("/gait_phase", Float32, queue_size=10)

        # Variable to store the gait phase
        self.current_phase = 0

    def callback(self, data):
        """
        Callback function to process the insole data and detect gait phases.
        Data is assumed to be a message of type InsoleData.
        """

        # Example of extracting data (pressure from the heel and toe areas)
        heel_pressure = data.heel
        toe_pressure = data.toe

        # Simple logic to define gait phases (you should adjust this based on your understanding of gait)
        if heel_pressure > some_threshold and toe_pressure < another_threshold:
            self.current_phase = 0  # Heel strike phase
        elif heel_pressure > mid_threshold and toe_pressure > mid_threshold:
            self.current_phase = 1  # Mid-stance phase
        elif heel_pressure < some_threshold and toe_pressure > some_threshold:
            self.current_phase = 2  # Toe-off phase

        # Publish the detected gait phase
        self.pub.publish(self.current_phase)

        rospy.loginfo(f"Gait Phase Detected: {self.current_phase}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        gait_node = GaitPhaseDetectionNode()
        gait_node.run()
    except rospy.ROSInterruptException:
        pass

