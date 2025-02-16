# TicTacToe
Using OpenCV to play Tic Tac Toe
## Overview
A fixed Axis security camera is used to interpret a tic tac toe grid, then an ABB IRB120 robotic arm is used to draw on the tic tac toe grid. It is important that both the robot and camera are in a fixed position. OpenCV is used to approximate the grid, and machine learning was used to interpret if a space is occupied with an X or an O. From there, a position in the image coordinate system is converted to fall in the robot's coordinate system. The robot then moves to the position and draws an o.
### Grid Interpretation

##### Calibration
To calibrate the system, a calibration image is used. The image has red dots at the corners of the robot's base, the center of which give the center. The image must also have a large checkerboard grid to find what the pixel to mm conversion of the image is. After getting this calibration image once, it should be stored. 

##### Grid Intersection Finding
Finding the grid in the frame is built on the assumption that the grid is green. First, using the OpenCV inRange function, a binary image is created, where anything green becomes white and everything else becomes black. The grid is then approximated using a Hough transform. Note that the Hough transform will likely find excess lines. Because of this, the intersections of lines that are nearly perpendicular are found. Points that are within a certain pixel distance of one another are grouped until only four points remain. These four points allow for other parts of the grid to be found, assuming it is symmetrical. 

##### Finding Center of Grid Positions
Given the four grid intersections, the approximate center of each position can be found by translating the intersection points. These points will later be important when moving the robot arm.

##### Checking Grid Occupancy
With the centers found, the occupancy of each position can be found using a convolutional neural network trained to identify X's and O's. First, a bounding box is created for a position, which is then cropped, rotated, and resized to be 28x28 pixels. The image must be binary. This new image can be passed into the ML model to be identified as an X or an O. The spot is considered empty if the number of white pixels in a particular crop is fewer than a threshold number. By repeating this process for each position, every approximate center can be associated with an X, O, or empty position.

##### Ordering the Grid
Finally, the grid must be properly organized, so win scenarios can be checked. This is accomplished by finding the closest position to the previous position starting with the center. Each previous position must be removed from the original list. The order of assignment becomes 4,1,2,5,8,7,6,3,0, meaning that you start at the center, go to the top row, middle position, then go clockwise around the grid from there.

### Selecting a Move
With the grid organized, a simple tic tac toe function selects a move on behalf of the robotic arm. It first checks the state of the grid to see if there is a win or a draw. If this is the case, the function announces that an end game scenario has occurred. Otherwise, a random open position, P, is selected for the robot to draw on.

### Robot Control

##### Conversion Between Camera and Robot Coordinate Systems
The camera image is in a left hand coordinate system, whereas the robot coordinate system is in a traditional right hand coordinate system. The camera coordinate system will be referred to as C, and the robot as R. To allow conversion from C to R, a series of transformations is necessary. First, the camera coordinate system must be converted to be a right hand coordinate system. This is accomplished by making all x values in C negative, which works because all the points in the image fall in the same quadrant. The points are then converted to mm distances based on the conversion values found in the calibration step. Then, the desired move, position P, must be translated by the robot's negative center coordinates in the image. By following the translation with a rotation of 90 degrees, the point is shifted to fall in the robot coordinate system. 

##### Sending a Move to the Robot
At this point, the move must be sent to the robot, so the robot can draw in the grid. The robot is initally moved to a writing position, where the pen is a few centimeters above the surface it is writing on. This requires that each joint of the robot arm is set to an angle. From this point on, all robot moves are linear paths in order to keep the pen in the correct position for drawing. A path is sent to have the robot hover over the selected position on the tic tac toe grid, and finally, the robot follows a precomputed path for drawing an O. After drawing, it goes to a home position so the whole process can be repeated again. 

ROS is used to both plan and execute robot commands using MoveIT. To communicate with the robot, a ROS service is created which sends moves to the robot through a socket. More details on this can be found on the ROS Wiki.



## Code Structure
### Classes
##### Camera
The Camera class handles image capturing, calibration, and cropping. It uses https requests to pull an images from the axis camera, then crops the image. Everytime this class is initialized, a path should be given to a calibration image, so pixel to mm conversions and the robot center can be found. To calibrate, call calibrate and pass in a path that links to a stored image. This class interacts directly with the Board class.

##### Model
The Model class handles loading the machine learning model from storage and, subsequently, identifying X's and O's. To identify an X or an O, call identify and pass in a 28x28 pixel image containing an X or an O, which will return a string. This class interacts directly with the Board class. 

##### Board
The Board class handles the tic tac toe grid recognition and comprehension as explained in the overview. This class is dependent on the Camera and Model classes. To make a board object, one must pass in a Model object, and an image from the camera class. To update the grid state, the Camera image must be updated then passed into the update_image method, a method in the Board class. Afterwards, the update method in Board must be called to identify the grid. This class interacts with the Game class to update the grid state.

##### TicTacToe
The TicTacToe class checks for win cases in the grid. A 2D list of length 9 is passed in and represents the state of the tic tac toe grid. Index 0 represents the top left corner, and subsequent indices are applied left to right, top to bottom. The last value of every sublist holds an 'x', 'o', or a ' '. To update the grid, this list must be passed into the update_positions method. Then, by calling the check_win a True value is returned if an endgame scenario has occured, a False value otherwise. This class interacts directly with the Game class.

##### Robot
The Robot class handles robot control and path planning using ROS. There are two options for control in this class. The first is to control the angle of each joint, which is used in moving the robot to known poses. The other option is path planning, where one can make the robot follow a cartesian path. This is used to draw shapes. To move the robot to a pose, the go_to_pose method can be called with either a string input or list input. A string will move the robot to a known position, and the list input will be a list of desired joint values. To plan and execute a path, the plan_and_execute method should be used. The input is a list of desired position increments. Also, if a string is entered insted of a list, it will follow a known path. This class interacts directly with the Game class to make a move.

ROS is used for all robot control. It speaks to the robot through a socket and ROS service. When a command is sent to the ROS service, that command is translated to something the robot can understand, then sent to the robot itself.

##### Game
The game class is where all the other classes integrate with one another. It handles the overall gameplay. In the game.py file, there is a main() function with can be called to run the game assuming that the robot and camera are properly set up. 

## Running the Code
1. Open three terminals.
2. Run ```roscore``` in the first.
3. Once the ```roscore``` is running, type ```roslaunch abb_irb120_moveit_config  moveit_planning_execution.launch sim:=false robot_ip:=<ROBOT_IP>``` in terminal 2.
4. Once RViz has opened, type ```python game.py``` in your project directory. The game should start running.


## Dependencies
##### ROS Installation
See http://wiki.ros.org/melodic/Installation/Ubuntu for Ubuntu installation. If you are using Windows, see https://janbernloehr.de/2017/06/10/ros-windows. 

##### OpenCV Installation
See https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/ for installation instructions.

##### Keras installation
```pip install keras```

## Support
Email clee4@olin.edu if any help is needed.

## ROS Resources
ROS Wiki: http://wiki.ros.org/
- General ROS tutorials and explanations

ROS Industrial Wiki: https://rosindustrial.org/
- Industrial robot control

ROS MoveIT Wiki: https://moveit.ros.org/
- Motion planning and visualization

ROS ABB Wiki: http://wiki.ros.org/abb
- ABB robot specific packages
