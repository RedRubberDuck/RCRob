import numpy as np
from matplotlib import pyplot as plt




def coeff(t):
    return [(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2,  t**3]

def main():
    print("Bezier line test")

    Px1 = 0
    Px2 = 0.2
    Px3 = 0.3
    Px4 = 0.4

    Py1 = 0
    Py2 = -0.5
    Py3 = 0.6
    Py4 = -0.8
    
    
    a1 = 0.0
    a2 = 0.25
    a3 = 0.75
    a4 = 1.0

    M = np.matrix([ coeff(a1), coeff(a2), coeff(a3), coeff(a4)])
    M_inv = np.linalg.inv(M)
    print(M,M_inv)

    Px = np.matrix([[Px1],[Px2],[Px3],[Px4]])
    Rx = M_inv * Px 
    Py = np.matrix([[Py1],[Py2],[Py3],[Py4]])
    Ry = M_inv * Py 

    print('X',Rx,'Y',Ry)




if __name__=="__main__":
    main()