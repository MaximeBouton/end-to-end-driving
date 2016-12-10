# end-to-end-driving
CS221-CS229 Class project. The objective is to predict steering command from raw camera input.

## Context

## Repository structure

**src**: host the main program \\
**plots**: contains some relevant illustrations\\

In the **src** folder, there are 2 subfolders:
- **classification**: CNN architecture to predict steering direction
- **regression** : 3 different CNN architectures to predict the steering angle


## Data

The dataset is provided by Udacity and contains a lot of information (3 different camera views, steering, brake and throttle reports). We used the 12 minutes part of the dataset and it can be found here: [dataset](https://github.com/udacity/self-driving-car/tree/master/datasets)

**Note**: when running the code be careful that your data files are in the right path.

*Extract the data*:
Need rosbag to be installed.
running `src/parse_bag.py` will convert the rosbag format into a csv file `steering.csv` containing the steering command and split the images into 3 folders (center_camera, left_camera and right_camera). The images are saved in jpeg format and the name of the images corresponds to the ROS time stamp.

**Note**: The time stamp of the steering command and of the images are not matching perfectly (different frequency). Some pre-processing will have to be done to map the time stamps.


## Dependencies

- Python 2.7
- ROS : ros-kinetic distribution
- OpenCV for python
- Keras (with TensorFlow backend)
- Matlab r2016b
