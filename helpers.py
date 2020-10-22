import math

def nothing(x):
    pass

def to_line(pts):
    """
    Converts set of points to slope intercept form

    pts: two points (x1,y1,x2,y2)
    """
    m = 0
    for x1,y1,x2,y2 in pts:
        # finds slope of line
        if x1-x2 == 0:
            m = float('inf')
        else:
            m = float(y1-y2)/float(x1-x2) 
        
        # finds x and y intercept of line
        yint = y1-(x1*m)
        xint = (x1+x2)/2
    return [round(m,2),round(yint,2),round(xint,2)]

def dist(p1,p2):
    """
    Finds distance between points
 
    p1, p2: point (x,y)

    returns float value for euclidean distance
    """
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

def avg(l1, l2):
    """
    Averages values at corresponding indices
    
    l1, l2: lists of values equal in length

    returns average list
    """
    for i in range(len(l1)):
        l1[i] = (l1[i]+l2[i])/2
    return l1

def midpt(p1,p2):
    """
    Finds midpoint of two points
 
    p1, p2: point (x,y)

    returns tuple containing the midpoint
    """
    return ((p1[0]+p2[0])/2.0,(p1[1]+p2[1])/2.0)

def find_closest(pt, pts):
    """
    Finds closest point in set of points

    pt: point to find closest distance to
    pts: set of pts to select from
    """
    min_dist = dist(pt, pts[0])
    index = 0
    for i in range(len(pts)):
        d = dist(pt, pts[i][:2])
        if d < min_dist:
            min_dist = d
            index = i
    
    return pts[index], index

if __name__ == "__main__":
    a = [[1,1],[1,2],[1,3],[2,2]]
    b = [1,1]
    print(find_closest(b,a))