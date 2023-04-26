#!/usr/bin/env python3

"""
This is the number detection node for exercise 5
"""

import numpy as np
import os
import math
import rospy
import time
import message_filters
import typing
import cv2
import yaml
import time
from cv_bridge import CvBridge

from statistics import mode

import rospkg
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Pose2DStamped, 
    LanePose, 
    WheelEncoderStamped, 
    WheelsCmdStamped, 
    Twist2DStamped,
    BoolStamped,
    VehicleCorners,
    SegmentList,
    LEDPattern,
    )
from duckietown_msgs.srv import SetCustomLEDPattern
from std_msgs.msg import Header, Float32, String, Float64MultiArray, Float32MultiArray, Int32
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Point32

import rosbag
from FC import NP_model

# Change this before executing
VERBOSE = 0
SIM = False

HSV_MASK_LOW = (80,45,70)
HSV_MASK_HIGH = (100,200,255)


class NumberDetectionNode(DTROS):
    """
    The Number Detection Node will subscribe to the camera and use a ML model to determine the number from an image with an AprilTag. 
    """
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(NumberDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh_name = os.environ["VEHICLE_NAME"]
        else:
            self.veh_name = "csc22935"

        self.rospack = rospkg.RosPack()

        self.model = None 
        self.load_model()
    

        # Static parameters
        self.update_freq = 30
        self.rate = rospy.Rate(self.update_freq)

        # Publishers
        self.pub_number_bb = rospy.Publisher(f"/{self.veh_name}/number_detection_node/image/compressed", CompressedImage, queue_size=1)
        self.pub_cropped_number = rospy.Publisher(f"/{self.veh_name}/number_detection_node/cropped_number/image/compressed", CompressedImage, queue_size=1)
        
        # Subscribers
        self.sub_images = rospy.Subscriber(f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_image, queue_size=1)


        self.log("Initialized")

    def cb_image(self, msg):
        #print("in cb_image")
        br = CvBridge()
        # Convert image to cv2
        raw_image = br.compressed_imgmsg_to_cv2(msg)
        copy_raw = raw_image

        # Start the timer
        start_time = time.time()

        # Mask everything but the post-it note
        image_hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
        image_mask = cv2.inRange(image_hsv, HSV_MASK_LOW, HSV_MASK_HIGH)

        # Get the contour of the post-it note
        contours, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours are found, return
        if len(contours) == 0:
            self.pub_processed_image(raw_image, self.pub_number_bb)
            return

        # Get the contour of the largest area        
        c = max(contours, key=cv2.contourArea)

        # Check if the area is big enough
        if cv2.contourArea(c) < 1000:
            self.pub_processed_image(raw_image, self.pub_number_bb)
            return

        # Get the bounding box of the contour
        number_bb = cv2.boundingRect(c)
        x = int(number_bb[0])
        y = int(number_bb[1])
        w = int(number_bb[2])
        h = int(number_bb[3])

        # Crop the bounding box to remove edge noise
        padding = int(min(h * 0.15, w * 0.15))

        # Crop the image
        number = raw_image[y + padding:y+h - padding, x + padding:x + w - padding]

        # Invert the bb image so the number is white and the background is black
        black_max = (90,125,125)
        black_min = (0,0,0)
        number_mask = cv2.inRange(number, black_min, black_max)
        # Resize the image to match the input size of the model
        number_mask = cv2.resize(number_mask, (28,28))
        self.pub_processed_image(number_mask, self.pub_cropped_number)
        input_vector = number_mask.reshape(1, 28 * 28)

        # Predict the number
        res_vector = self.model.predict(input_vector)
        number = np.argmax(res_vector)

        # Stop the timer
        end_time = time.time()
        # Determine the duration
        duration = (end_time - start_time) * 1000
        duration_str = "{:.3f}".format(duration) + " ms"
        
        # Add a bounding box and text to the image
        cv2.rectangle(raw_image,(x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(raw_image, str(number), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,12), 2)
        cv2.putText(raw_image, str(duration_str), (x + 30 ,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255,255), 2)

        self.pub_processed_image(raw_image, self.pub_number_bb)
            

        self.rate.sleep()

    def pub_processed_image(self, image, publisher):
        compressed_image = CompressedImage()
        compressed_image.header.stamp = rospy.Time.now()
        compressed_image.format = "jpeg"
        compressed_image.data = np.array(cv2.imencode('.jpg',image)[1]).tostring()

        publisher.publish(compressed_image)


    def load_model(self):
        # Get the filepath to the weight
        model_file_folder = self.rospack.get_path('number_detection') + '/config/model_weights.npy'

        # Import the weights
        weight_dict = np.load(model_file_folder, allow_pickle=True).item()
        self.model = NP_model(weight_dict)


if __name__ == '__main__':
    node = NumberDetectionNode(node_name='robot_follower_node')
    # Keep it spinning to keep the node alive
    # main loop
    rospy.spin()
