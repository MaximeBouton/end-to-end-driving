import rosbag
import cv2
import os
import csv

from cv_bridge import CvBridge


cc_bag = rosbag.Bag('../../data/center_cam.bag')
lc_bag = rosbag.Bag('../../data/left_cam.bag')
rc_bag = rosbag.Bag('../../data/right_cam.bag')
st_bag = rosbag.Bag('../../data/vehicle.bag')

##########  write steering in csv file ###################

steering =  open('../../data/steering2.csv','w')
headers = ['seq','time_stamp','steering_wheel_angle','steering_wheel_angle_cmd','steering_wheel_torque','speed']
writer = csv.DictWriter(steering, fieldnames=headers)
writer.writeheader()

for topics, msg, t in st_bag.read_messages(topics =['/vehicle/steering_report']):
    row = dict()
    row['seq']     = str(msg.header.seq)
    row['time_stamp'] = str(msg.header.stamp)
    row['steering_wheel_angle'] = str(msg.steering_wheel_angle)
    row['steering_wheel_angle_cmd'] = str(msg.steering_wheel_angle_cmd)
    row['steering_wheel_torque'] = str(msg.steering_wheel_torque)
    row['speed'] = str(msg.speed)
    writer.writerow(row)
    print topics,msg.header.seq

steering.close()

############    convert images to .jpeg files #############

# initialize cv bridge
bridge = CvBridge()

cur_path = os.getcwd()

### CENTER CAMERA
try:
    os.chdir('../../data/center_camera2') # directory where you save the images
except:
    os.mkdir('../../data/center_camera2')
    os.chdir('../../data/center_camera2') # directory where you save the images

for topics, msg, t in cc_bag.read_messages(topics = ['/center_camera/image_color/compressed']):
    img_name = str(msg.header.stamp)+'.jpeg'
    cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding = 'passthrough')
    cv2.imwrite(img_name,cv_img)
    print topics,msg.header.seq

os.chdir(cur_path) # go back to current directory

### LEFT CAMERA
try:
    os.chdir('../../data/left_camera2') # directory where you save the images
except:
    os.mkdir('../../data/left_camera2')
    os.chdir('../../data/left_camera2') # directory where you save the images

for topics, msg, t in lc_bag.read_messages(topics = ['/left_camera/image_color/compressed']):
    img_name = str(msg.header.stamp)+'.jpeg'
    cv_img = bridge.compressed_imgmsg_to_cv2(msg,  desired_encoding = 'passthrough')
    cv2.imwrite(img_name,cv_img)
    print topics,msg.header.seq

os.chdir(cur_path) # go back to current directory

### RIGHT CAMERA
try:
    os.chdir('../../data/right_camera2') # directory where you save the images
except:
    os.mkdir('../../data/right_camera2')
    os.chdir('../../data/right_camera2') # directory where you save the images

for topics, msg, t in rc_bag.read_messages(topics = ['/right_camera/image_color/compressed']):
    img_name = str(msg.header.stamp)+'.jpeg'
    cv_img = bridge.compressed_imgmsg_to_cv2(msg,  desired_encoding = 'passthrough')
    cv2.imwrite(img_name,cv_img)
    print topics,msg.header.seq

os.chdir(cur_path) # go back to current directory




