#!/usr/bin/env python3

import rospy
import socket
import struct
from std_msgs.msg import Float32MultiArray

# Import compiled protobuf messages
import common_pb2
import service_pb2

def parse_moticon_data(data):
    """Parse received protobuf data."""
    message = service_pb2.DataMessage()
    message.ParseFromString(data)
    return message

def tcp_server():
    rospy.init_node('moticon_insoles', anonymous=True)
    pub = rospy.Publisher('moticon_data', Float32MultiArray, queue_size=10)

    host = "0.0.0.0"  # Listen on all interfaces
    port = 5000       # Match with OpenGo App settings

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    rospy.loginfo("Waiting for connection from OpenGo Mobile App...")

    conn, addr = server_socket.accept()
    rospy.loginfo(f"Connected to {addr}")

    while not rospy.is_shutdown():
        # Read message length (first 2 bytes, big-endian)
        length_prefix = conn.recv(2)
        if not length_prefix:
            break

        message_length = struct.unpack('>H', length_prefix)[0]
        data = conn.recv(message_length)

        if data:
            parsed_data = parse_moticon_data(data)
            sensor_values = list(parsed_data.values)  # Extract sensor data
            msg = Float32MultiArray(data=sensor_values)
            pub.publish(msg)

    conn.close()

if __name__ == '__main__':
    try:
        tcp_server()
    except rospy.ROSInterruptException:
        pass
