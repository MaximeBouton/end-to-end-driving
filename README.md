# end-to-end-driving
CS221-CS229 Class project. The objective is to predict steering command from raw camera input.

## Context

## Repository structure

*src*: host the main programs
*proposal*: host the project proposal

## Data

Need rosbag to be installed.
running `src/parse_bag.py' will convert the rosbag format into a csv file `steering.csv` containing the steering command and split the images into 3 folders (center_camera, left_camera and right_camera). The images are saved in jpeg format and the name of the images corresponds to the ROS time stamp.

**Note**: The time stamp of the steering command and of the images are not matching perfectly (different frequency). Some pre-processing will have to be done to map the time stamps. 


## Dependency

- Python 2.7
- ROS : ros-kinetic distribution
- OpenCV for python

