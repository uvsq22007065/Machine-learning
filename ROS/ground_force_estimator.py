#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray, Int16
from scipy.signal import butter, filtfilt
import numpy as np
from collections import deque

class vGRFEstimator:
    def __init__(self):
        #ROS variables
        rospy.init_node('ground_reaction_estimator', anonymous=True)

        self.foot_imu_sub = rospy.Subscriber('/insole_data', Int32MultiArray, self.insole_data_callback)

        self.vGRF_pub = rospy.Publisher('/vGRF', Int16, queue_size=2)
        self.test_pub = rospy.Publisher('/test', Int32MultiArray, queue_size=2)

        self.noise_threshold = 400
        
        # Filter parameters
        self.cutoff_freq = 2  # Cutoff frequency in Hz
        self.Fs = 100  # Sample rate in Hz
        self.order = 2
        
        # Design Butterworth filter
        self.b, self.a = self.butter_lowpass(self.cutoff_freq, self.Fs, self.order)

        # Buffer for incoming data
        self.buffer_size = 10  # Should be greater than the filter order
        self.data_buffer = deque(maxlen=self.buffer_size)  # Stores the last N samples

    def butter_lowpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_filter(self, data):
        # Apply the filter to the entire data
        filtered_data = filtfilt(self.b, self.a, data)
        return filtered_data


    def insole_data_callback(self, msg):
        current_data = np.array(msg.data) 
        self.data_buffer.append(current_data) 
        
        if len(self.data_buffer) == self.buffer_size: 
            buffer_array = np.array(self.data_buffer)
            # Apply the filter to each position (column) in the buffer
            filtered_data = np.apply_along_axis(self.apply_filter, 0, buffer_array)  # Apply along columns
            self.process_data(np.round(filtered_data[-1]).astype(int))

            int_filtered_data = np.round(filtered_data[-1]).astype(int) 
            # Prepare the Int32MultiArray for publishing
            int_array_msg = Int32MultiArray(data=int_filtered_data.astype(np.int32).tolist())
            self.test_pub.publish(int_array_msg)


    def process_data(self, data):

        # Group insole data in regions: heel, mid and tip
        heel_data = np.sum(data[11:15])
        if heel_data < self.noise_threshold:
            heel_data = 0
        mid_data = np.sum(data[5:10])
        if mid_data < self.noise_threshold:
            mid_data = 0
        tip_data = np.sum(data[0:4])
        if tip_data < self.noise_threshold:
            tip_data = 0
        
        # Estimate the ground reaction force as the maximum value
        vGRF = max(heel_data,mid_data,tip_data)
        
        self.vGRF_pub.publish(vGRF)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        estimator = vGRFEstimator()
        estimator.run()
    except rospy.ROSInterruptException:
        pass
