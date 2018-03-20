import numpy as np

class LineEstimatorBasedPolynom:
    def __init__(self,laneWidthCm,pxpcm,windowSize):
        self.windowSize = windowSize
        self.laneWidthPx = laneWidthCm * pxpcm

    def estimateLine(self,polynomlines):
        maxLen = None
        maxKey = None
        nonLineKey = []
        lines = []
        for key in polynomlines.keys():
            if (maxLen is None or polynomlines[key].line or maxLen < len(polynomlines[key].line)) and len(polynomlines[key].line) > 5 :
                maxLen = len(polynomlines[key].line)
                maxKey = key
            elif len(polynomlines[key].line) < 2 :
                nonLineKey.append(key)
        

        if maxLen is None :
            print("It wasn't detect any line")
            return polynomlines
        # print(nonLineKey,maxKey)

        largestLine = polynomlines[maxKey].line
        for key in nonLineKey:
            keyDiff = key - maxKey
            polynomlines = self.generatePoint(polynomlines,largestLine,key,keyDiff)
            # lines.append(line)
        return polynomlines

    def generatePoint(self,polynomlines,largestLine,key,keyDiff):
        line = []

        for index in range(len(largestLine)-1):
            pointI = largestLine[index]
            pointI1 = largestLine[index-1]

            # dY = pointI1[1] - pointI[1]
            # dX = pointI1[0] - pointI[0]
            vecI_I1 = pointI1-pointI
            
            # dis = np.sqrt(dY**2 + dX**2)
            dis = abs(vecI_I1)
            rate = self.laneWidthPx*abs(keyDiff) / dis

            if( dis.imag*keyDiff > 0 ):
                vecI_I1*=-1
                # dX = -1 * dX
                # dY = -1 * dY
            transVec = vecI_I1*complex(0,1)
            newPoint = pointI1 - transVec * rate
            # newPointY = pointI1[1] + vecI_I1.real * rate
            if(self.windowSize[0]>= newPoint.real and newPoint.real>=0 and self.windowSize[1]>=newPoint.imag and newPoint.imag>0):
                line.append(newPoint)
        # line = sorted(line,key=operator.itemgetter(1))
        polynomlines[key].line = line
        return polynomlines

