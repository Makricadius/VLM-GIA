import numpy as np

def rad(angle):
    return angle/180*np.pi
def deg(radians):
    return radians/np.pi*180

def rotate(x0,y0,alpha):
    x1 = x0*np.cos(alpha)-y0*np.sin(alpha)
    y1 = x0*np.sin(alpha)+y0*np.cos(alpha)
    return x1,y1