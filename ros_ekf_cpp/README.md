# Ros and C++ based Tracking algorithm

# Usage

'''
catkin make
'''

'''
roslaunch ros_ekf_cpp.launch
'''

# Visualization

1. Green box : Detected object from CenterPoint
2. Blue box : A bounding box after update process of EKF
3. white box  : A bounding box predicted n frame after position

# Description
1. Input : jsk_recognition_msgs::BoundingBoxArray (About 11hz, 0,09sec)
2. Output : Estimated n frames after bounding box, after update process bounding box
3. /odom topic is used for coordinate transform (Local -> Global, Global -> Local)

# Warning

1. Please restart Roslaunch when Bag file is finished
