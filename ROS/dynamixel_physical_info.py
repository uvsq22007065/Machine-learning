#!/usr/bin/env python3

import rospy
from ankle_exoskeleton.msg import DynamixelStatusList
# from ankle_exoskeleton.msg import DynamixelStatus
from ankle_exoskeleton.msg import DynamixelStatusListPhysical
from ankle_exoskeleton.msg import DynamixelStatusPhysical

class DynamixelStatusRepublisherNode:
    def __init__(self):
        rospy.init_node('dynamixel_status_republisher_node', anonymous=True)

        namespace = rospy.get_param('~namespace', '') 

        if namespace:
            namespace = namespace + "/"

        self.sub_topic = namespace + 'dynamixel_motor/status'
        self.pub_topic = namespace + 'dynamixel_motor/status_physical'

        self.sub = rospy.Subscriber(self.sub_topic, DynamixelStatusList, self.dynamixel_status_callback)
        self.pub = rospy.Publisher(self.pub_topic, DynamixelStatusListPhysical, queue_size=1)

        
    def dynamixel_status_callback(self, msg):
        modified_statuses = []
        
        for status in msg.dynamixel_status:
            modified_status = DynamixelStatusPhysical()
            modified_status.name = status.name
            modified_status.id = status.id

            modified_status.present_pwm_pctg = float("{0:.1f}".format(status.present_pwm * 0.113))  # [%]
            modified_status.present_current_amp = float("{0:.3f}".format(status.present_current * 3.36 / 1000))  #[A]
            modified_status.present_position_deg = float("{0:.1f}".format(status.present_position*0.088))    #[deg]
            modified_status.present_velocity_rpm = float("{0:.1f}".format(status.present_velocity*0.229))    #[rpm] 

            modified_status.present_input_voltage_v = float("{0:.1f}".format(status.present_input_voltage*0.1, 3))    #[V]
            modified_status.present_temperature_c = status.present_temperature    #[°C]

            
            modified_statuses.append(modified_status)
        
        republished_msg = DynamixelStatusListPhysical()
        republished_msg.dynamixel_status_physical = modified_statuses
        self.pub.publish(republished_msg)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DynamixelStatusRepublisherNode()
        node.run()
    except rospy.ROSInterruptException:
        pass