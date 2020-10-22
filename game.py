#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import camera as cam
import model
import os
from copy import copy
from robot import RobotArm
from board import Board
from tictactoe import TicTacToe

def rotate_grid(grid):
    remap = [6,3,0,7,4,1,8,5,2]
    new_grid = []

    for i in remap:
        new_grid.append(grid[i])
    return new_grid

def reflect_grid(grid, axis):
    remap = []
    if axis is 'x':
        remap = [6,7,8,3,4,5,0,1,2]
    else:
        remap = [2,1,0,5,4,3,8,7,6]

    new_grid = []
    for i in remap:
        new_grid.append(grid[i])

    return new_grid

class Game:
    def __init__(self, show=True):
        self.show = show

        self.mycam = cam.Camera()

        mlmodel = model.XOModel(os.path.join('storage', 'model.json'), 
                              os.path.join('storage', 'model.h5'))

        self.computer_first = True
        self.p1good = model.TicTacModel(os.path.join('storage', 'p1_good.h5'))
        self.p2good = model.TicTacModel(os.path.join('storage', 'p2_good.h5'))

        self.p1learning = model.TicTacModel(os.path.join('storage', 'p1_learning.h5'))
        self.p2learning = model.TicTacModel(os.path.join('storage', 'p2_learning.h5'))

        self.board = Board(self.mycam.frame, mlmodel)
        self.tictac = TicTacToe()
        self.grabby = RobotArm()
 
        self.o_count = 0
        self.x_count = 0

    def change_difficulty(self):
        """
        Changes the difficulty of the game
        """
        print('Enter a difficulty from 1 to 3. 1 being the easiest and 3 being the hardest.')
        try:
            self.difficulty = int(raw_input())
        except:
            self.difficulty = 2

        print('Difficulty is set to: ', self.difficulty)    

    def get_grid(self):
        """
        Pulls grid from tic tac toe object

        returns current state of the grid
        """
        return self.tictac.grid

    def count(self, shape, grid):
        """
        Counts the number of spaces occupied with a certain shape

        shape: string of shape to be counted
        grid: list containing state of all grid positions

        returns int containing total number of given shape
        """
        num = 0
        for pos in grid:
            if pos[-1] == shape:
                num+=1
        return num

    def cam_to_tic(self):
        """Converts cv information to tic tac toe grid information"""
        grid = copy(self.board.positions)
        grid_untouched = copy(grid)
        

        self.tictac.update_positions(grid)

    def make_move(self, move):
        """
        Given a move and grid state, the robot is given a series of commands to make it move

        move: int containing the index of desired move
        """
        try:
            # robot moves to writing position
            self.grabby.go_to_pose("writing")

            # camera coordinate is converted to robot coordinate
            robot_pos = cam.cam_to_robo(move, self.mycam.robot_center, 
                                        self.mycam.convW, self.mycam.convH) + [None]

            # robot path is planned and displayed
            plan, _ = self.grabby.point_to_path([robot_pos])
            self.grabby.display_trajectory(plan)

            # path is executed and an x is drawn
            self.grabby.execute_plan(plan)
            rospy.sleep(.2)
            self.grabby.plan_and_execute("o")

            # after making its move, robot goes to home position
            rospy.sleep(.1)
            self.grabby.go_to_pose("home")

        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            pass

    def draw_win(self, pos):
        """
        Connects the three winning shapes of a win case

        pos: list of positions to connect
        """
        robot_pos1 = cam.cam_to_robo(pos[0], self.mycam.robot_center, 
                                        self.mycam.convW, self.mycam.convH)
        robot_pos2 = cam.cam_to_robo(pos[1], self.mycam.robot_center, 
                                        self.mycam.convW, self.mycam.convH)
        
        try:
            # robot moves to writing position
            self.grabby.go_to_pose("writing")

            # robot path is planned and displayed
            plan, _ = self.grabby.point_to_path([robot_pos1+[None]])
            self.grabby.display_trajectory(plan)
            self.grabby.execute_plan(plan)
            rospy.sleep(.1)

            plan, _ = self.grabby.plan_cartesian_path([[0,0,-.09]])
            self.grabby.display_trajectory(plan)
            self.grabby.execute_plan(plan)
            rospy.sleep(.1)

            plan, _ = self.grabby.point_to_path([robot_pos2+[None]])
            self.grabby.display_trajectory(plan)
            self.grabby.execute_plan(plan)
            rospy.sleep(.1)

            plan, _ = self.grabby.plan_cartesian_path([[0,0,.09]])
            self.grabby.display_trajectory(plan)
            self.grabby.execute_plan(plan)
            rospy.sleep(.1)
            
            self.grabby.go_to_pose("home")
        
        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            pass

    def update(self, check=False):
        """
        Updates position coordinates and state of the tic tac toe grid

        show: whether or not to show the frame that was captured
        check: check if grid interpretation makes sense
        """
        while True:
            # updates frame and state of tictactoe grid
            self.mycam.get_frame()
            self.board.update_image(self.mycam.frame)
            self.board.update(draw=self.show)

            self.cam_to_tic()

            # operator checks that the robot is reading the image appropriately
            print(self.tictac)
            print('Type GO if camera image is good to move the robot')
            # shows cv image
            if self.show:
                cv2.imshow("frame", self.board.frame)
                cv2.waitKey(30)
            move = raw_input()
            if move == 'go':
                break
    
    def do_turn(self):
        win, pos = self.tictac.check_win()
        # checks for win
        if win:
            self.draw_win(pos)
            
            return False
        else:
            move = -1
            # selects next move for robot
            if self.difficulty == 1:
                move = self.tictac.select_move()
            elif self.difficulty == 2:
                if self.computer_first:
                    move = self.p1learning.pick_move(self.tictac.grid)
                else:
                    move = self.p2learning.pick_move(self.tictac.grid)
            else:
                if self.computer_first:
                    move = self.p1good.pick_move(self.tictac.grid)
                else:
                    move = self.p2good.pick_move(self.tictac.grid)
            print(move)
            self.make_move(self.tictac.grid[move])

        return True

    def play_game(self):
        """Plays game of tic tac toe when called"""

        print('Press ENTER to move robot.')
        raw_input() 
        self.grabby.go_to_pose('all zero')
        print('Please remove the cap. Press ENTER when finished.')
        raw_input()
        self.grabby.go_to_pose("home")

        self.change_difficulty()

        cont = True
        new_game = True
        while cont:
            if new_game:
                print("If you would like to go first, make your move then press ENTER. Otherwise, just press ENTER.")
                raw_input()

                self.update()

                if self.count(' ', self.tictac.grid) == 9:
                    self.computer_first = True
                else:
                    self.computer_first = False
                
                cont = self.do_turn()

                new_game = False


            print('Type DONE to quit or press ENTER if your move has been made.')
            if raw_input() == 'done':
                break
            self.update()
            cont = self.do_turn()

            if cont == False:
                print('Would you like to play another game? If so, please replace Tic Tac Toe grid then type YES.')
                x = raw_input()
                if x == 'yes':
                    
                    self.change_difficulty()

                    cont = True
                    new_game = True

        self.grabby.go_to_pose('all zero')
        print('Please replace the cap. Press ENTER when finished.')
        raw_input()
        self.grabby.go_to_pose("home")


def main(show=False):
    game = Game(show=show)
    print(game)

    game.play_game()



if __name__ == "__main__":
    main(True)