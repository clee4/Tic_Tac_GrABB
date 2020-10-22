import os
import cv2
import math
import json
import numpy as np
import camera as cam

from PIL import Image
from helpers import *

class Board:
    """
    This class finds the position of a tic tac toe grid in frame using OpenCV.
    """
    def __init__(self, frame, model, calibrate=False, draw=False):
        path = os.path.join('storage','thresholds.json')
        with open(path) as json_file:
            data = json.load(json_file)
            self.grid_color = (tuple(data['min_green']),
                               tuple(data['max_green']))
            self.xo_thresh = tuple(data['xo_thresh'])
        
        self.update_image(frame)    
        self.height, self.width = self.frame.shape[:2]

        if calibrate:
            self.calibrate()

        self.model = model

        # sets initial variables and frame
        self.update(draw)

    def calibrate(self):
        """Allows the user to tweak colors to better pick out specific colors"""
        # converts frame to HSV
        frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)

        # calibrates grid color
        self.grid_color = cam.calibrate(frame,'Grid',self.grid_color)

        # outputs color ranges for later use
        print('grid_color', self.grid_color)

    def update_image(self, frame):
        """Updates the image being used to check the state of the board"""
        self.frame = frame
        self.untouched_frame = frame.copy()
        self.get_grid()

    def get_grid(self):
        """
        Masks out everything but the grid based on color
        
        min_thresh: minimum color threshold for the mask
        max_thresh: maximum color threshold for the mask
        The color value of the grid in hsv should fall between min and max
        """
        self.grid_mask, self.grid_only = cam.color_mask(self.frame, dilation=2, erosion=2, 
                                                               min_thresh=self.grid_color[0], 
                                                               max_thresh=self.grid_color[1])

    
    def find_lines(self, img=-1, minLineLength=25, maxLineGap=15):
        """
        Uses Hough Transform to find all the lines in the image
        
        minLineLength: minimum line length to be considered line
        maxLineGap: the max gap in a line for it to still be considered a line
        """
        if img == -1:
            img = self.grid_mask

        # finds lines in the image for hough lines detection
        self.lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength,maxLineGap)

    def segment_to_ab(self, lines=-1, color=(0,255,0), draw=False):
        """
        Converts lines to a slope and y intercept
        
        lines: lines to convert to ab
            if -1 uses self.lines
        draw: whether or not the lines should be drawn on the frame
        color: color of lines to be drawn

        returns lines in ab form
        """
        ab = []

        if lines == -1:
            lines = self.lines

        for line in lines:
            # adds slope, xint, and y int to list
            ab.append(to_line(line))

        # draws lines on frame
        if draw:
            cam.draw_lines(self.frame, ab, color)

        return ab

    def find_intersections(self, lines=-1, slope_thresh=3, color=(255,0,0), draw=False):
        """
        Finds where each line intersects
        
        lines: lines to find intersections between
               if lines is -1, use lines in self.ab
        color: color to draw in
        draw: whether or not to draw on the frame

        returns list of intersections
        """
        pts = []

        if lines == -1:
            lines = self.ab

        for i in range(len(lines)):
            # line to check intersections with
            line1 = lines[i]
            # iterates through all other lines and finds intersection points
            for j in range(i+1, len(lines)):
                line2 = lines[j]

                # checks for the case of vertical and similar lines
                if (abs(line1[0])>slope_thresh and abs(line2[0])>slope_thresh) or (abs(line1[0]-line2[0]) < 1):
                    pass
                else:
                    # handles case if line1 is vertical
                    if line1[0] > slope_thresh:
                        x = line1[2]
                        y = line2[0]*x + line2[1]
                    # handles case if line2 is vertical
                    elif line2[0] > slope_thresh:
                        x = line2[2]
                        y = line1[0]*x + line1[1]
                    # handles generic case and solves for system of equations
                    else:
                        x = (line2[1]-line1[1])/(line1[0]-line2[0])
                        y = line1[0]*x + line1[1]

                    # ignores points outside of the frame
                    if x < 0 or y < 0 or x > self.width or y > self.height:
                        pass
                    else:
                        # adds intersection point to point list
                        pts.append([int(x),int(y)])
        # draws intersections
        if draw:
            self.frame = cam.draw_points(self.frame, pts, color)

        return pts

    def group_points(self, pts=-1, dist_thresh=30, color=(0,0,0), draw=False):
        """
        Groups point within a distance threshold

        dist_thresh threshold for points to be grouped
        color: color for points to be drawn in. Default is black
        draw: whether or not to draw on the frame

        returns grouped points in ascending order based on their x values
        """
        if pts == -1:
            pts = self.pts

        grouped_points = []

        # iterates through all the points
        while len(pts) > 0:
            pt = pts[0]

            # compares each point with every other point
            i = 0
            while i < len(pts):
                # if two points are within a certain distance of each other
                if dist(pt, pts[i]) < dist_thresh:
                    # average points and keep track
                    pt = avg(pt,pts[i])
                    # remove point form list
                    pts.pop(i)
                else:
                    i+=1
            
            grouped_points.append(tuple(pt))

        # draws circles for all the points
        if draw:
            cam.draw_points(self.frame, self.pts, color)

        # sort points based on their x coordinate
        return sorted(grouped_points, key=lambda x: x[0])

    def find_grid_lines(self, color=(0,0,255), draw=False):
        """
        Finds tic tac toe grid based on points

        color: color for lines to be drawn in. Default is green
        draw: whether or not to draw on the frame
        """
        grid_lines = []
        pts = self.pts

        # finds closest two points to pt to form outline of center of tic tac toe grid
        for pt in self.pts:
            # sorts points based on their distance from pt
            temp = sorted(pts, key=lambda x: dist(pt,x))

            # takes the middle two points and appends them
            # the middle two points form the edges of the center of the grid
            grid_lines = grid_lines + [[pt+temp[1]],[pt+temp[2]]]

        self.ab = self.segment_to_ab(grid_lines)

        i = 0
        # removes duplicate lines from self.ab
        while i < len(self.ab):
            # checks for number of instances
            if self.ab.count(self.ab[i]) > 1:
                self.ab.pop(i)
                i-=1
            i+=1
        
        # draws grid lines
        if draw:
            self.frame = cam.draw_lines(self.frame, self.ab, color)
            self.frame = cam.draw_points(self.frame, self.pts, (0,255,0))

    def find_positions(self, color=(0,255,0), draw=False):
        """
        Finds the center of each square in tic tac toe grid
        
        color: color to draw centers of squares in
        draw: whether or not to draw on the frame
        """
        # sorts lines based on their slope
        lines = sorted(self.ab, key=lambda x: abs(x[0]))
        
        # averages similar lines
        self.avg_lines = [avg(lines[0],lines[1]),avg(lines[2],lines[3])]
        # finds center of grid
        self.center = self.find_intersections(lines=self.avg_lines)

        # computes distance between grid intersections and center
        distances = []
        for pt in self.pts:
            x_dist = pt[0]-self.center[0][0]
            y_dist = pt[1]-self.center[0][1]
            distances.append([x_dist,y_dist])
        
        # uses distances to move known grid intersections into center of each space
        positions = []
        for distance in distances:
            for pt in self.pts:
                x = pt[0]+distance[0]
                y = pt[1]+distance[1]
                positions.append([x,y])
        
        # groups similar points to get rid of duplicates
        self.positions = self.group_points(positions, 40)

        # draws circles for positions
        if draw:
            self.frame = cam.draw_points(self.frame, self.positions, color)
    
    def sort_positions(self):
        """Sorts positions in grid for checking win case"""
        sorted_positions = [[]]*9

        # assigns index four to center position
        current, index = find_closest(self.center[0], self.positions)
        sorted_positions[4] = self.positions.pop(index)
        
        # continuously finds the closest point
        current, index = find_closest(current, self.positions)
        sorted_positions[1] = self.positions.pop(index)

        current, index = find_closest(current, self.positions)
        sorted_positions[2] = self.positions.pop(index)

        current, index = find_closest(current, self.positions)
        sorted_positions[5] = self.positions.pop(index)
        
        current, index = find_closest(current, self.positions)
        sorted_positions[8] = self.positions.pop(index)

        current, index = find_closest(current, self.positions)
        sorted_positions[7] = self.positions.pop(index)

        current, index = find_closest(current, self.positions)
        sorted_positions[6] = self.positions.pop(index)

        current, index = find_closest(current, self.positions)
        sorted_positions[3] = self.positions.pop(index)

        current, index = find_closest(current, self.positions)
        sorted_positions[0] = self.positions.pop(index)

        self.positions = sorted_positions

    def crop_position(self, index, binary_img, width=28, height=28):
        """
        Crops grid positions to be used for machine learning

        index: position index to be cropped
        binary_img: image to crop from
        width: pixel width of outputted image
        height: pixel height of outputted image

        returns cropped image
        """
        if binary_img == -1:
            binary_img = self.untouched_frame.copy()
            binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(binary_img,self.xo_thresh[0],
                                          self.xo_thresh[1],cv2.THRESH_BINARY_INV)
            binary_img = cv2.dilate(binary_img, kernel = np.ones((3,3), np.uint8), iterations=1)

        # finds distance from center position to position
        x_dist = self.center[0][0]-self.positions[index][0]
        y_dist = self.center[0][1]-self.positions[index][1]

        # uses distances to make bounding box for positions
        updated_pts = []
        for x, y in self.pts:
            updated_pts = updated_pts + [[[x-x_dist,y-y_dist]]]
        
        # sorts the updated points based on their distance fromthe first point
        updated_pts = sorted(updated_pts, key=lambda x: dist(self.positions[index][:2], x[0]))

        # switches values to ensure list is in order
        updated_pts[2], updated_pts[3] = (updated_pts[3], updated_pts[2])
        
        updated_pts = np.asarray(updated_pts)

        # crops image
        img_crop, _ = cam.crop_rect(binary_img, updated_pts)

        # cv2.imshow('asdf', img_crop)
        # cv2.waitKey(0)
        
        return cv2.resize(img_crop,(width,height))

    def check_occupancy(self, img=-1, min_white=60):
        """
        Checks if space is occupied and what is occupying it

        img: image to crop from
        min_white: number of white pixels for the spot to be occupied
        """
        # iterates through all the positions
        for i in range(len(self.positions)):
            self.positions[i] = list(self.positions[i])
            # crops position
            position = self.crop_position(i, img)
            

            # counts the number of white pixels
            num_white = cv2.countNonZero(position)

            # checks if space is occupied or empty
            if num_white > min_white:
                self.positions[i].append(self.model.identify(position, show=True))
            else:
                self.positions[i].append(' ')

    def update(self, num_tries=50, draw=False):
        """
        Updates frame with new tic tac toe grid position
        
        num_tries: The maximum number of tries to find intersections
        draw: Whether or not all found points should be drawn
        """
        self.asdf = self.frame.copy()
        for _ in range(num_tries):
            try:
                self.find_lines()
                self.ab = self.segment_to_ab(draw=draw)
                self.pts = self.find_intersections(self.ab, draw=draw)
                self.pts = self.group_points(draw=draw)
                self.find_grid_lines(draw=draw)   
            except:
                print('[ERROR]: No lines found')
            
            try:
                if len(self.pts) == 4:
                    self.find_positions(draw=draw)
                    self.check_occupancy()
                    self.sort_positions()
                    break
            except:
                print('[ERROR]: No intersections found')
        
        
if __name__ == '__main__':
    import model
    
    mycam = cam.Camera()
    model = model.XOModel(os.path.join('storage', 'model.json'), 
                          os.path.join('storage', 'model.h5'))
    board = Board(mycam.frame, model, calibrate=False, draw=True)
    
    while 1:
        mycam.get_frame()
        
        board.update_image(mycam.frame)

        board.update(draw=True)
        cv2.imshow('frame', board.frame)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('c'):
            board.calibrate()
        