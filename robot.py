#!/usr/bin/env python

import shapes
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

class RobotArm():
    def __init__(self, grid_size=.048, group='manipulator'):
        """
        Initializes robotic arm object to control arm and its movement using
        MoveIt commander

        grid_size: the size of each space in tic tac toe grid
        
        """
        # scales grid size down slightly to ensure that drawings fit within
        grid_size = grid_size*.9

        # intializes moveit commander and move robot note in ros
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_robot', anonymous=True)

        # creates robot object for getting robot state
        self.robot = moveit_commander.RobotCommander()

        # creates object for creating objects in the environment
        self.scene = moveit_commander.PlanningSceneInterface

        # creates move group for robot arm based on the groups name
        # this name is set when creating the robot arm
        self.move_group = moveit_commander.MoveGroupCommander(group)

        # creates object for displaying paths in rviz
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                        moveit_msgs.msg.DisplayTrajectory,
                                                        queue_size=20)

        # creates dictionaries that store default poses and paths
        self.poses = {'all zero' : (0,0,0,0,0,0),
                      'home'     : (0,0,-1.5,0,0,0),
                      'writing'  : (0,.5,.5,0,.55,0)}
        self.paths = {'o' : shapes.circle_path(grid_size/2,grid_size/20),
                      'x' : shapes.x_path(grid_size)}

    def __str__(self):
        """Returns string for robot state"""
        return "==== Printing robot state ====\n%s\n" % (self.get_state())

    def get_state(self):
        """Returns robot state"""
        return self.robot.get_current_state()
    
    def get_joint_values(self):
        """Returns robot's joint values"""
        return self.move_group.get_current_joint_values()

    def go_to_pose(self, pose):
        """
        Moves robot to specific joint angles in radians

        pose: list containing desired joint angles from base link
        """
        # if pose is a string it will pull a pose from path
        if isinstance(pose, str):
            pose = self.poses[pose]

        # sets joint values to desired positions
        joint_goal = self.move_group.get_current_joint_values()
        joint_goal[0:len(pose)] = pose

        # moves robot to desired joint positions
        self.move_group.go(joint_goal, wait=True)

        # stops the robot and prevents any additional movements
        self.move_group.stop()

    def plan_cartesian_path(self, path, scale=1):
        """
        Creates a cartesian path that the robot can follow

        path: list of increments in the x y and z directions
        scale: percent value for scaling
        """
        waypoints = []

        # gets robot's current pose
        wpose = self.move_group.get_current_pose().pose

        # iterates through increments in path and adds them to path
        for target in path:
            wpose.position.x += scale*target[0]
            wpose.position.y += scale*target[1]
            wpose.position.z += scale*target[2]
            waypoints.append(copy.deepcopy(wpose))

        # plans robot path
        (plan, fraction) = self.move_group.compute_cartesian_path(
                                        waypoints, 0.01, 0.0)
        
        return plan, fraction

    def point_to_path(self, targets):
        """
        Creates path for moving the robot to a desired target position 
        but only moves robot on xy plane

        target: target robot position
        """
        waypoints = []
        wpose = self.move_group.get_current_pose().pose

        # sets desired position of robot to target values
        for x,y,z in targets:
            if not x is None:
                wpose.position.x = x
            if not y is None:
                wpose.position.y = y
            if not z is None:
                wpose.position.x = z
            waypoints.append(copy.deepcopy(wpose))

        # plans path for the target
        (plan, fraction) = self.move_group.compute_cartesian_path(
                                        waypoints, 0.01, 0.0)

        return plan, fraction

    def display_trajectory(self, plan):
        """
        Displays planned path in rviz

        plan: path to display in rviz
        """
        # creates move_it message
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        # sets the start position of trajectory to be displayed
        display_trajectory.trajectory_start = self.robot.get_current_state()
        # adds remainder of trajectory
        display_trajectory.trajectory.append(plan)
        
        # publishes path to topic for rviz to read
        self.display_trajectory_publisher.publish(display_trajectory)

    def execute_plan(self, plan):
        """
        Sends movement commands to specified move group
        
        plan: plan to be executed
        """
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()

    def plan_and_execute(self, path, scale=1):
        """
        Plans and execute desired path

        path: list of increments in the x y and z directions
        scale: percent value for scaling
        """
        if isinstance(path, str):
            path = self.paths[path]

        cartesian_plan, _ = self.plan_cartesian_path(path,scale)
        self.display_trajectory(cartesian_plan)
        self.execute_plan(cartesian_plan)

    def teleop(self):
        """Allows user to control robot through keyboard commands"""
        while True:
            print("Make a move!\ne, q: +/- x\nd, a: +/- y\nw,s: +/- x\nh: home\nr: writing position\no: draw circle\nx: draw x\np: quit")
            move = raw_input()
            if move is 'p':
                break
            elif move is 'h':
                self.go_to_pose("home")
            elif move is 'r':
                self.go_to_pose("writing")
            elif move is 'e':
                self.plan_and_execute([[.01,0,0]])
            elif move is 'q':
                self.plan_and_execute([[-.01,0,0]])
            elif move is 'd':
                self.plan_and_execute([[0,.01,0]])
            elif move is 'a':
                self.plan_and_execute([[0,-.01,0]])
            elif move is 'w':
                self.plan_and_execute([[0,0,.01]])
            elif move is 's':
                self.plan_and_execute([[0,0,-.01]])
            elif move is 'x':
                self.plan_and_execute("x")
            rospy.sleep(.1)
        self.go_to_pose("home")

if __name__ == '__main__':
    try:
        grabby = RobotArm()
        grabby.teleop()

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
