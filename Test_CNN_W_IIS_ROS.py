#!/usr/bin/env python3

import rospy
import matlab.engine
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from std_msgs.msg import Float32, String
import time

def start_matlab_engine():
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    return eng

def process_data(eng):
    # Run MATLAB scripts to extract force data and gait vector for training and testing
    eng.eval('dynamics_estimator_hysteresis', nargout=0)
    force_data_train = eng.workspace['force_data']
    gait_vector_train = eng.workspace['gait_vector']
    gait_phases_train = eng.workspace['gait_phases']
    ankle_angles_filt_train = eng.workspace['ankle_angles_filt']

    eng.eval('dynamics_estimator_hysteresis_test', nargout=0)
    force_data_test = eng.workspace['force_data']
    gait_vector_test = eng.workspace['gait_vector']
    gait_phases_test = eng.workspace['gait_phases']
    ankle_angles_filt_test = eng.workspace['ankle_angles_filt_test']

    return (force_data_train, gait_vector_train, gait_phases_train, ankle_angles_filt_train,
            force_data_test, gait_vector_test, gait_phases_test, ankle_angles_filt_test)

def train_cnn_model(X_train, y_train):
    # Build the CNN model
    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # Assuming 4 phases of gait

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=32)

    return model

def run_model(eng):
    # Start the timer
    start_time = time.time()

    # Get data from MATLAB engine
    (force_data_train, gait_vector_train, gait_phases_train, ankle_angles_filt_train,
     force_data_test, gait_vector_test, gait_phases_test, ankle_angles_filt_test) = process_data(eng)

    # Preprocess and interpolate data (similar to your original code)
    # -- INSERT YOUR EXISTING DATA PROCESSING LOGIC HERE --

    # Train CNN model
    model = train_cnn_model(X_combined_reshaped_train, y_seq_train)

    # Predict on the test set
    y_pred = model.predict(X_combined_reshaped_test)
    y_pred = y_pred.argmax(axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_seq_test, y_pred)
    rospy.loginfo(f"CNN Model Accuracy: {accuracy * 100:.2f}%")

    # Stop the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    rospy.loginfo(f"Execution time: {elapsed_time} seconds")

    # Publish results (optional, depending on your application)
    pub_accuracy = rospy.Publisher('cnn_model_accuracy', Float32, queue_size=10)
    pub_accuracy.publish(accuracy * 100)

    return accuracy, y_pred

def cnn_ros_node():
    # Initialize ROS node
    rospy.init_node('cnn_gait_phase_detection', anonymous=True)

    # Start MATLAB engine
    eng = start_matlab_engine()

    try:
        # Main logic for processing data and training the model
        accuracy, y_pred = run_model(eng)
    except Exception as e:
        rospy.logerr(f"Error running the model: {e}")
    finally:
        # Stop MATLAB engine
        eng.quit()

if __name__ == '__main__':
    try:
        cnn_ros_node()
    except rospy.ROSInterruptException:
        pass
