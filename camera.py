import cv2
import numpy as np
import os
import requests
import glob
import imutils
import json

from math import cos, sin, pi
from helpers import *
from PIL import Image
from io import BytesIO
from requests.auth import HTTPDigestAuth

BIG_NUMBER = 9999999999

def calibrate(frame, name, current=-1):
    """
    Allows for color thresholds to be calibrated
    
    frame: the image being used for calibration
    name: the name of what is being calibrated for
    current: current hsv range being used for thresholding

    returns tuple containing two tuples that hold hsv thresholds
    """
    if current == -1:
        current = ((0,0,0),(0,0,0))

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('H min',name,current[0][0],179,nothing)
    cv2.createTrackbar('S min',name,current[0][1],255,nothing)
    cv2.createTrackbar('V min',name,current[0][2],255,nothing)

    cv2.createTrackbar('H max',name,current[1][0],179,nothing)
    cv2.createTrackbar('S max',name,current[1][1],255,nothing)
    cv2.createTrackbar('V max',name,current[1][2],255,nothing)

    while True:
        hmin = cv2.getTrackbarPos('H min',name)
        smin = cv2.getTrackbarPos('S min',name)
        vmin = cv2.getTrackbarPos('V min',name)
        min_thresh = (hmin,smin,vmin)

        hmax = cv2.getTrackbarPos('H max',name)
        smax = cv2.getTrackbarPos('S max',name)
        vmax = cv2.getTrackbarPos('V max',name)
        max_thresh = (hmax,smax,vmax)

        mask = cv2.inRange(frame,min_thresh,max_thresh)

        cv2.imshow(name, cv2.resize(mask, (1024,576)))

        key = cv2.waitKey(1)

        if key == ord('q'):
            cv2.destroyWindow(name)
            return current
        elif key == ord('s'):
            cv2.destroyWindow(name)
            return (min_thresh,max_thresh)

def correct_camera_warp(frame, points_x=7, points_y=5):
    """
    Fixes warp in camera image from lens

    points_x: number interior points of checkerboard in frame in the x direction
    points_y: number interior points of checkerboard in frame in the y direction
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((points_y*points_x,3), np.float32)
    objp[:,:2] = np.mgrid[0:points_y,0:points_x].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # converts image to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (points_y,points_x),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    h,  w = frame.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    print(roi)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    return dst

def crop_rect(img,pts):
    """
    Crops rectangle based on points

    pts: pts to be included in cropped space
    """
    rect = cv2.minAreaRect(pts)
    
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot

def get_conversion(img, points_h=5, points_w=7, show=False):
    """
    Finds pixel to mm conversion

    points_h: number interior points of checkerboard in frame in the y direction
    points_w: number interior points of checkerboard in frame in the x direction
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (points_h,points_w),None)
    corner1 = list(corners[0])[0]
    corner2 = list(corners[1])[0]
    corner3 = list(corners[6])[0]
    x1,y1 = corner1
    x2,y2 = corner2
    x3,y3 = corner3
    x1 = int(x1)
    x2 = int(x2)
    x3 = int(x3)
    y1 = int(y1)
    y2 = int(y2)
    y3 = int(y3)

    # Find the dimensions of a checkerboard square in pixels
    pxW = dist(corner2,corner1)
    pxH = dist(corner2,corner3)

    if show:
        print("Checker Square Corners: (",x1,",",y1,")","(",x2,",",y2, ")")
        print("Checker Square Dimensions W:", pxW, "H:",pxH)

    # Compare pixel dimensions to mm dimensions to find conversion factor
    convW = 42.0/pxW
    convH = 42.0/pxH

    return (convW,convH)

def get_robot_center(img, min_area=1000, max_area=3000, show=False):
    """
    Finds center of robot base in frame using contours

    img: to find robot center in
    min_area: minimum contour area to be considered
    max_area: maximum contour area to be considered
    show: whether or not to show cv frame
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = (0,100,100)
    upper_red = (6,255,255)
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = cv2.erode(mask,(5,5),iterations = 3)
    # find contours in the thresholded image
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
	                        cv2.CHAIN_APPROX_SIMPLE)
    points = []

    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > min_area and cv2.contourArea(c) < max_area:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append([cX,cY])
            # draw the contour and center of the shape on the image
            cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
            cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(img, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    center = midpt(points[0], points[1])
    if show:
        cv2.circle(img, center, 7, (255, 255, 255), -1)
        cv2.putText(img, "robot_center", (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('img',img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            pass
    
    return center

def cam_to_robo(point, robot_center, convW, convH):
    """
    Converts from camera to robot coordinate system
    
    point: point to put in robot coordinate system
    robot_center: pixel coordinates of robot in image
    convW: pixel to mm conversion factor in x direction
    convH: pixel to mm conversion factor in y direction
    """
    # convert point to mm and stores as vector
    point = np.matrix([[-point[0]*convW],
                       [point[1]*convH], 
                       [1]])
    robot_center = [-robot_center[0]*convW, robot_center[1]*convH]

    # creates rotation matrix
    theta = pi/2
    rotate = np.matrix([[cos(theta), sin(theta), 0],
                        [-sin(theta), cos(theta), 0],
                        [0, 0, 1]])

    # creates translation matrix
    translate = np.matrix([[1, 0, -robot_center[0]],
                           [0, 1, -robot_center[1]],
                           [0, 0, 1]])
    
    # translates then rotates the point
    # point is now in robot coordinate system
    robot_point = np.dot(rotate, np.dot(translate, point))
    
    return [round(robot_point[0],1)/1000, round(robot_point[1],1)/1000]

def draw_lines(frame, lines, color):
    """
    Draws lines on the frame

    lines: the list of lines that need to be drawn
            each line should be in the form [m, yint, xint]
    color: the color for the lines to be drawn in
    """
    for m, yint, xint in lines:
        # accounts for case where the slope is close to infinity
        if abs(m) < BIG_NUMBER:
            x1 = 0
            x2 = len(frame[0])
            y1 = x1*m + yint
            y2 = x2*m + yint 
        else:
            x1 = xint
            x2 = xint
            y1 = 0
            y2 = 100000
        frame = cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,3)
    
    return frame

def draw_points(frame, points, color):
    """
    Draws points on the frame

    points: the list of points that need to be drawn
            each point should be in the form [x,y]
    color: the color for the pointss to be drawn in
    """
    for pt in points:
        frame = cv2.circle(frame,(int(pt[0]),int(pt[1])),3,color,-1)
    
    return frame

def draw_contours(frame, contours, color):
    """
    Draws contours on the frame

    lines: the list of contours found using OpenCV find contours
    color: the color for the lines to be drawn in
    """
    frame = cv2.drawContours(frame, contours, -1, color, 1, 8)

    return frame

def color_mask(frame, kernel=-1, dilation=2, erosion=2, 
               min_thresh=(0,0,0), max_thresh=(255,255,255)):
    """
    Makes a mask based on a certain color range in hsv
    by default every color is included  

    frame: image to make mask from
    kernel: kernel to be used for morphologies
    dilation: number of dilates to do
    erosion: number of erodes to do
    min_thresh: min color threshold
    max_thresh: max color threshold

    returns mask and frame with only colors in range
    """
    if kernel == -1:
        kernel = np.ones((5,5), np.uint8)
    
    # converts image to hsv 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #creates mask based on range for the color red
    mask = cv2.inRange(hsv, min_thresh, max_thresh)

    # removes holes in image
    mask = cv2.dilate(mask, kernel, iterations=dilation)
    mask = cv2.erode(mask, kernel, iterations=erosion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # combines mask with original image to be left with only a specific color
    coloronly = cv2.bitwise_and(frame, frame, mask=mask)
    
    # returns mask and updated frame
    return (mask, coloronly)

class Camera():
    def __init__(self, calibration_img='calibration.png'):
        # to be used for morphologies
        self.kernel = np.ones((5,5), np.uint8)

        # must have first run source command to get credentials
        self.url = os.environ.get('CAM_HOST')
        self.user = os.environ.get('CAM_USER_NAME')
        self.password = os.environ.get('CAM_PASS')

        path = os.path.join('storage','thresholds.json')
        with open(path) as json_file:
            data = json.load(json_file)
            self.crop = data['crop']

        # sets intial frame
        self.get_frame()

        self.calibrate(calibration_img, show=False)

        # finds frame resolution of camera
        self.width, self.height = self.frame.shape[:2]
    
    def calibrate(self, calibration_img, show=False):
        """
        Based on calibration image, pixel to mm conversion and robot
        center are found

        calibration_img: path to calibration image
        show: boolean value to pring out found values
        """
        img = cv2.imread(calibration_img,1)
        self.convW, self.convH = get_conversion(img)
        self.robot_center = get_robot_center(img, show=show)

        if show:
            print(self.convW, self.convH)
            print("Robot center: ", self.robot_center)

    def get_frame(self):
        """Pulls frame from tethered camera using http address"""
        url_response = requests.get(self.url, auth=HTTPDigestAuth(self.user,self.password))
        i = Image.open(BytesIO(url_response.content))
        self.frame = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR)
        self.frame = cv2.GaussianBlur(self.frame, (5,5), 0)

        if self.crop == -1:
            width, height = self.frame.shape[:2]
            self.crop = (0, 0, width, height)

        self.frame = self.frame[self.crop[1]:self.crop[3], 
                                self.crop[0]:self.crop[2]]
        return self.frame

    def update(self):
        self.get_frame()


if __name__ == "__main__":
    cam = Camera()
    # cv2.imwrite('calibration.png', cam.get_frame())
    # img = cv2.imread('calibration.png',1)
    # frame = correct_camera_warp(img)
    # cv2.imshow('unwarped', frame)
    # cv2.imshow('warped', img)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     pass
    # point = [500,500]
    # cam = Camera()
    # cam.frame = draw_points(cam.frame, [point], (0,0,255))

    # cv2.imshow('asdf', cam.frame)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     pass
            
    # print(cam.convW, cam.convH, cam.robot_center)

    # print(cam_to_robo(point,cam.robot_center,cam.convW, cam.convH))
    # cv2.destroyAllWindows()