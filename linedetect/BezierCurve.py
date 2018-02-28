from numpy import conjugate as conj
from numpy import power 
from numpy import arctan
import math
import time 

import numpy as np
from matplotlib import pyplot as plt

class CubicBezierCurve:
    def __init__(self,f_A=complex(0,1),f_B=complex(0,2),f_C=complex(1,2),f_D=complex(1,3)):
        self.A=f_A
        self.B=f_B
        self.C=f_C
        self.D=f_D
        self.S=[f_A,3*(f_B-f_A),3*(f_C-2*f_B+f_A),(f_D-3*f_C+3*f_B-f_A)]
        self.dS=[self.S[1],self.S[2]*2,self.S[3]*3]
        self.ddS=[self.dS[1],self.dS[2]*2]
    
    
    def S_func(self,f_t):
        return self.S[0]+self.S[1]*f_t+self.S[2]*f_t**2+self.S[3]*f_t**3
    
    def dS_func(self,f_t):
        return self.dS[0]+self.dS[1]*f_t+self.dS[2]*f_t**2
        
    def ddS_func(self,f_t):
        return self.ddS[0]+self.ddS[1]*f_t

    def K(self,f_t):
            dS=self.dS_func(f_t)
            ddS=self.ddS_func(f_t)
            denum=power(dS*conj(dS),3.0/2.0).real
            num=((dS*conj(ddS)-conj(dS)*ddS)/(-2j)).real    
            return num/denum
    def vel(self,f_t):
        dS_r=self.dS_func(f_t)
        return sqrt((dS_r*conj(dS_r)).real)

class PolynomFunc:
    def __init__(self,coeff):
        self.coeff = coeff
    def __call__(self,t):
        res = 0
        for i in range(len(self.coeff)):
            res += self.coeff[i]*np.power(t,i)
        return res

class BezierCurve:
    def __init__(self,points):
        self.order = len(points)-1 
        self.controll = points
        self.polynom = PolynomFunc(BezierCurve.getCoeff(points))
    
    @staticmethod
    def getCoeff(points):
        n =  len(points) - 1 
        n_fac = math.factorial(n)
        coeff = []
        for j in range(0, n +1):
            coeffJ = 0 

            for i in range(0 , j+1):
                coeffJ += power(-1,i+j)*points[i]/ (math.factorial(i)*math.factorial(j-i))
            
            coeffJ *= n_fac / math.factorial(n-j) 
            coeff . append(coeffJ)
        return coeff

def coeff(t):
    return [(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2,  t**3]

def process(Point0,Point33,Point66,Point1):
    M = np.matrix([coeff(0),coeff(0.33),coeff(0.66),coeff(1)])
    M_inv = np.linalg.inv(M)

    Points = np.matrix([[Point0],[Point33],[Point66],[Point1]])
    ResControlPoint = M_inv * Points
    print(ResControlPoint)
    return [ResControlPoint[0,0], ResControlPoint[1,0], ResControlPoint[2,0], ResControlPoint[3,0]]

def getPoint(Points):
    nrPoint = len(Points)
    arcLength = 0
    for i in range(0,nrPoint-1):
        P1 = Points[i]
        P2 = Points[i+1]
        dis = np.abs(P1-P2)
        arcLength += dis

    j33 = j66 = 0
    l33 = l66 = 0
    for i in range(0,nrPoint-1):
        P1 = Points[i]
        P2 = Points[i+1]
        dis = np.abs(P1-P2)

        if l33 + dis > arcLength/3 and l33 < arcLength/3:
            j33 = i
            l33 += dis
        else:
            l33 += dis
    
        if l66 + dis > arcLength/3*2 and l66 < arcLength/3*2:
            j66 = i
            l66 += dis
        else:
            l66 += dis

    print(j33,j66)
    return j33,j66
    


def tupleListToComplexList(points):
    newComplexPoints = []
    for point in points:
        newComplexPoints.append(complex(point[0],point[1]))
    return newComplexPoints

def ComplexListToTupleList(points):
    newComplexPoints = []
    for point in points:
        newComplexPoints.append((int(np.real(point)),int(np.imag(point))))
    return newComplexPoints


def XYToTuple(X,Y):
    newPoints = []
    for x,y in zip(X,Y):
        newPoints.append((int(x),int(y)))
    return newPoints

if __name__ == "__main__":
    p1 = complex(0,0);p2 = complex(2,0);p3 = complex(3,0); p4 = complex(3,2); p5 = complex (3,3)
    points = [p1,p2,p3,p4,p5]

    bezc1 = BezierCurve(points)
    t = np.linspace(0,1.0,100)
    P = bezc1.polynom(t)
    X = np.real(P)
    Y = np.imag(P)

    # j33,j66 = getPoint(P)
    
    # Point33 = P[j33]
    # # Point33 = P[index33]
    # Point66 = P[j66]
    # # Point66 = P[index66]
    # newPoints = process(points[0],Point33,Point66,points[-1])

    # becz2 = BezierCurve(newPoints)
    # P2 = becz2.polynom(t)
    # # print(P2)


    minX = np.min(X)
    maxX = np.max(X)

    
    t1 = time.time()
    coeffX = np.polyfit(X,Y,9)
    # coeffY = np.polyfit(t,Y,4)

    t2 = time.time()

    print("Duration",t2-t1)
    polynom = np.poly1d(coeffX)
    # polynomY = np.poly1d(coeffY)
    # xx = polynomX(XX)
    yy = polynom(X)



    # plt.plot(np.real(points),np.imag(points),'or')
    # plt.plot(np.real([Point33,Point66]),np.imag([Point33,Point66]),'o')
    # plt.plot(np.real(newPoints),np.imag(newPoints),'og')
    plt.plot(X,Y)
    plt.plot(X,yy,'--r')
    # plt.plot(np.real(P2),np.imag(P2),'--r')
    plt.show()