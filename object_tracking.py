import cv2
import sys
import time
import numpy as np
from helpers import midpt, dist

def nothing(x):
    pass

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


def create_mask(frame, min=(0,30,77), max=(10,208,215)):
    img = cv2.GaussianBlur(frame.copy(),(5,5),0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img, min, max)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask
    
def lowest_chicken(mask, thresh=-1):
    if thresh == -1:
        thresh = .0022*mask.shape[0]*mask.shape[1]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
	                                  cv2.CHAIN_APPROX_SIMPLE)

    good_contour = None
    for contour in contours:
        contour = cv2.convexHull(contour, False)
        M = cv2.moments(contour)
        y = int(M['m01']/M['m00'])
        if cv2.contourArea(contour) > thresh and y > mask.shape[0]*.8:
            good_contour = contour
            break

    return cv2.boundingRect(good_contour)

def box(bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    x, y = midpt(p1,p2)
    return (p1, p2, (int(x), int(y)))

def main():
    # Set up tracker.
    tracker = cv2.TrackerKCF_create()

    # Read video
    video = cv2.VideoCapture("chicken.mp4")
    prev_time = time.time()
    _, frame = video.read()

    mask = create_mask(frame)
    bbox = lowest_chicken(mask)
    _, _, prev_centroid = box(bbox)

    # start OpenCV object tracker using the supplied bounding box
    # coordinates, then start the FPS throughput estimator as well
    tracker.init(frame, bbox)
    count = 0
    while True:
        # Read a new frame
        cur_time = time.time()
        ok, frame = video.read()
        if not ok:
            break
        
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1, p2, center = box(bbox)
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
            cv2.circle(frame, center, 5, (0,255,0), -1)
            
            # dx = center[0]-prev_centroid[0]
            # dy = center[1]-prev_centroid[1]
            dt = .033

            # vx = int(dx/dt)
            # vy = int(dy/dt)
            if count == 0:
                v = int(dist(center,prev_centroid)/dt)
                velocity = v
                count+=1
            else:
                vnew = int(dist(center,prev_centroid)/dt)
                v = int((vnew+v)/2)
                count+=1

            prev_centroid = center
        else:
            # Tracking failure
            tracker = cv2.TrackerKCF_create()
            mask = create_mask(frame)
            bbox = lowest_chicken(mask)
            tracker.init(frame, bbox)
            _, _, prev_centroid = box(bbox)

        if count % 20 == 0:
            velocity = v
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Velocity: '+str(velocity)+' pixel/s',(30,30), font, 1,(0,255,0),2,cv2.LINE_AA)

        # Display result
        cv2.imshow("Tracking", cv2.resize(frame,(1024,576)))
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

if __name__ == '__main__' :
    main()