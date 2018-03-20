import numpy as np
from PolynomLine import *



def inWindow(x,y,windowSize):
    return ( (x > 0 and x < windowSize[0]) and (y > 0 and y < windowSize[1]))

class LaneMiddleGenerator:
    def __init__(self,laneWidthCm,pxpcm,windowSize,polyDeg):
        self.laneWidthPx =  laneWidthCm * pxpcm
        print("LaneDetector ",self.laneWidthPx)
        self.windowSize = windowSize
        self.nrPoint = 20
        self.pxpcm = pxpcm
        self.stepY = self.windowSize[1] / self.nrPoint
        self.polyDeg = polyDeg
    
    def generateLine (self,polynomlines,middleLine = None):
        line = []
        if(len(polynomlines)>=2):

            middleLineKey = 1.5
            limitMinY = self.windowSize[1]
            limitMaxY = 0
            for key in polynomlines.keys():
                if key == 0:
                    continue
                if polynomlines[key].polynom is None or len(polynomlines[key].line)< polynomlines[key].polyDeg:
                    continue
                polynomline = polynomlines[key]
                dis = middleLineKey - key
                for index in range(self.nrPoint):
                    pointY = self.stepY * index
                    if (pointY > polynomline.lineInterval[0] and pointY < polynomline.lineInterval[1]):
                        
                        pointX = polynomline.polynom(pointY)
                        dP = polynomline.dPolynom(pointY)
                        dP_ = -1 / dP
                        dY = self.laneWidthPx*dis  / np.sqrt(dP_**2+1)
                        dX = dY * dP_
                        if(dX*dis < 0):
                            dX = -1*dX
                            dY = -1*dY
                        if inWindow(pointX+dX,pointY+dY,self.windowSize):
                            line.append(complex(pointX+dX,pointY+dY))
                
                if limitMinY is None or limitMinY > polynomline.lineInterval[0] :
                    limitMinY = polynomline.lineInterval[0]  
                if limitMaxY is None or limitMaxY < polynomline.lineInterval[1] :
                    limitMaxY = polynomline.lineInterval[1]  
            
            if middleLine is None:
                middleLine = PolynomLine(self.polyDeg)
            
            middleLine.estimatePolynom(line)
            if middleLine.polynom is None:
                return None

            newline = []
            for index in range(self.nrPoint):
                pointY = self.stepY * index
                if limitMaxY > pointY and limitMinY < pointY:
                    pointX = middleLine.polynom(pointY)
                    newline.append(complex(pointX,pointY))

            middleLine .line = newline   
            return middleLine
        else:
            return None
