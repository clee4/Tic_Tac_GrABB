import math
import numpy as np

def circle_path(r=1, step=.1, z_offset=.09):
    """
    Creates path that outlines a circle 

    r: radius of desired circle
    step: the number of points to be stored
    z_offset: height offset at which circle needs to be drawn

    returns list of points
    """
    points = []

    # creates list of x_values evenly stepped
    x_values = np.linspace(0,2*r,int(r*2/step)).tolist()
    
    # finds points for the top half of the circle
    for x in x_values: 
        points.append([x, math.sqrt(r**2 - (x-r)**2)])
    # finds points for the bottom half of the circle
    for x in reversed(x_values[0:len(x_values)-1]):  
        points.append([x, -math.sqrt(r**2 - (x-r)**2)])

    
    targets = []
    prev_point = points[0]
    # converts coordinates of circle to increments for a path
    for point in points[1:]:
        x = point[0] - prev_point[0]
        y = point[1] - prev_point[1]
        targets.append([x,y,0])

        prev_point = point

    return [[-r,0,0],[0,0,-z_offset]]+targets+[[0,0,z_offset]]

def x_path(width,z_offset=.09):
    """
    Creates path that draws a square x

    width: width of the x to be drawn
    z_offset: height offset at which circle needs to be drawn

    returns list of points
    """
    # finds four corners of the x
    p0 = [0,0]
    p1 = [p0[0]+width,p0[1]+width]
    p2 = [p0[0]+width,p0[1]]
    p3 = [p0[0],p0[1]+width]

    # aligns robot with corner of x
    targets = []
    x = p1[0]-p0[0]
    y = p1[1]-p0[1]
    targets.append([x,y,0])

    # draws one diagonal of the x
    targets.append([0,0,z_offset])
    x = p2[0]-p1[0]
    y = p2[1]-p1[1]
    targets.append([x,y,0])

    # draws other diagonal of the x
    targets.append([0,0,-z_offset])
    x = p3[0]-p2[0]
    y = p3[1]-p2[1]
    targets.append([x,y,0])

    return [[-width/2,-width/2,0],[0,0,-z_offset]]+targets+[[0,0,z_offset]]

if __name__ == "__main__":
    print(circle_path())