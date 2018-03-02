import cv2
import numpy as np
import operator
import scipy.optimize



def distance(PointI,PointJ):
    return np.sqrt((PointI[0]-PointJ[0])**2 + (PointI[1]-PointJ[1])**2)

# def getTheLargestLine(lines):
#         largestLineIndex = None
#         largestLineLength = None
#         nrLine = len(lines)  
#         for i in range(nrLine):
#             if largestLineLength is None or largestLineLength < len(lines[i]):
#                 largestLineLength = len(lines[i])
#                 largestLineIndex = i
#         return largestLineIndex


# ## It implementing a method with check the lines relation
# class LaneLineEstimator:
#     def __init__(self,laneWidth,pxpcm):
#         self.laneWidthPx = laneWidth * pxpcm
#         self.pxpcm = pxpcm
    

#     def estimateLine(self,lines):
#         Index = self.getTheLargestLine(lines)
#         # Index = 0

#         # newLines = []
#         L1 = []
#         L2 = []
#         L3 = []
#         L4 = []
#         curLine = lines[Index]
#         nrLine = len(curLine)

#         for i in range(0,nrLine-2):
#             PointI = curLine[i]
#             PointI1 = curLine[i+1]
#             PointI2 = curLine[i+2]

#             dX1 = PointI1[0] - PointI[0]
#             dY1 = PointI1[1] - PointI[1]

#             dX2 = PointI2[0] - PointI1[0]
#             dY2 = PointI2[1] - PointI1[1]

#             dX = (dX1 + dX2) / 2
#             dY = (dY1 + dY2) / 2



#             dist = np.sqrt(dX **2 + dY **2 )
#             dist1 = np.sqrt(dX1 **2 + dY1 **2 )
            
#             rate = self.laneWidthPx / dist  
#             rate1 = self.laneWidthPx / dist1  

#             newPoint1X = int(PointI1[0] + dY*rate)
#             newPoint1Y = int(PointI1[1] - dX*rate)

#             newPoint2X = int(PointI1[0] - dY*rate)
#             newPoint2Y = int(PointI1[1] + dX*rate)
            
#             newPoint1X_p = int(PointI1[0] + dY1*rate1)
#             newPoint1Y_p = int(PointI1[1] - dX1*rate1)

#             newPoint2X_p = int(PointI1[0] - dY1*rate1)
#             newPoint2Y_p = int(PointI1[1] + dX1*rate1)

#             L1.append((newPoint1X,newPoint1Y))
#             L2.append((newPoint2X,newPoint2Y))
            
#             L3.append((newPoint1X_p,newPoint1Y_p))
#             L4.append((newPoint2X_p,newPoint2Y_p))

#         return [L1,L2,L3,L4]


        
        



class LaneVerifierBasedDistance:
    def __init__(self,laneWidth,pxpcm,errorCm =7):

        self.laneWidthPX = laneWidth*pxpcm
        self.pxpcm = pxpcm
        self.errorLaneWidthPX = errorCm * pxpcm 
    def checkDistanceBetweenLines(self,lineI, lineJ):
        
        nrPointI = len(lineI)
        nrPointJ = len(lineJ)

        if (nrPointI < nrPointJ):
            lineRef = lineI
            lineEx = lineJ
        else:
            lineRef = lineJ
            lineEx = lineI

        sum_of_distance = 0
        for i in range(len(lineRef)):
            minDis =  None
            for j in range(len(lineEx)):
                dist  = distance(lineRef[i],lineEx[j])

                if minDis is None or minDis > dist:
                    minDis = dist

            sum_of_distance  += minDis
        mean_of_distance = sum_of_distance/ len(lineRef)
        # print(mean_of_distance)
        return np.abs(mean_of_distance - self.laneWidthPX) < self.errorLaneWidthPX

    
    def checkLane(self,lines):
        nrLines = len(lines)

        laneIndex = [None] * nrLines
        Lanes = [] 
        nrLane = 0

        if (nrLines <= 1):
            print('Detected nr. lines:',nrLines)
            return lines
        for curI in range(0,nrLines-1):
            lineCur = lines[curI]
            for prevI in range(curI+1,nrLines):
                linePrev = lines[prevI]
                isPair = self.checkDistanceBetweenLines(lineCur,linePrev)
                
                if ( isPair):
                    if(laneIndex[curI] is None and laneIndex[prevI] is None):
                        laneIndex[curI] = nrLane
                        laneIndex[prevI] = nrLane
                        Lanes.append([curI,prevI])
                        nrLane += 1
                    elif(laneIndex[curI] is None):
                        laneIndex[curI] = laneIndex[prevI]
                        Lanes[ laneIndex[prevI] ].append(curI)
                    elif (laneIndex[prevI] is None):
                        laneIndex[prevI] = laneIndex[curI]
                        Lanes[ laneIndex[curI] ].append(prevI)


        if(nrLane==1):
            print("Line",Lanes[0])
            newLines = []
            for index in Lanes[0]:
                newLines.append(lines[index])
            return newLines
        else:
            print("Distance based lane grouping failedd!",nrLane)
        return None 
        
class LineConvter:
    
    
    @staticmethod
    def TupleList2ComplexList(line):
        newline = [] 
        for point in line:
            newline.append(complex(line[0],line[1]))
        return newline
    
    @staticmethod
    def TupleList2ArrayList(line):
        return np.array(line)

    @staticmethod
    def XY2TupleList(f_x,f_y):
        newline = [] 
        for x,y in zip(f_x,f_y):
            newline.append((int(x),int(y)))
        return newline


def LineOrderCheck(polynomLine_dic,imagesize):
    lineTestPos_dic = {}
    Y = imagesize[1]/2
    for key in polynomLine_dic:
        polynomLine = polynomLine_dic[key]
        if polynomLine.polynom is None:
            continue
        X = polynomLine.polynom(Y)
        lineTestPos_dic[key]=X
    sorted_line = sorted(lineTestPos_dic.items(), key=operator.itemgetter(1))

    newPolynomLine_dic = {}
    for index in range(len(sorted_line)):
        key,pointX = sorted_line[index]
        # print(index,key)
        newPolynomLine_dic[index] = polynomLine_dic[key]
    
    return newPolynomLine_dic



class PolynomLine:
    def __init__(self,polyDeg):
        self.polyDeg = polyDeg
        self.polynom = None
        self.dPolynom = None 
        self.line = []
        self.lineInterval = None
    def estimatePolynom(self,line):
        if len(line) <= self.polyDeg:
            print("Warming: Not enough to estimate the polynom")
            return
        l_point_a = np.array(line)
        l_y = l_point_a[:,1]
        l_x = l_point_a[:,0]

        

        if self.polynom  is not None:
            # popt,pcov = scipy.optimize.curve_fit(PolynomLine.poly,l_y,l_x,p0=self.polynom.coef)
            # print((self.polynom.coef-popt))
            coeff1 = np.polyfit(l_y,l_x,self.polyDeg) 
            error = abs(self.polynom.coef-coeff1)
            if (error[0] < 0.001 and error[1] < 5 and error[2]<500):
                coeff = coeff1
            else:
                print ("error:",error)
                coeff = self.polynom.coef
            # coeff = (coeff*0.5 + self.polynom.coef*0.9)
        else:
            coeff = np.polyfit(l_y,l_x,self.polyDeg) 
        self.polynom = np.poly1d(coeff)
        self.dPolynom = self.polynom.deriv()
        self.line = line
        self.lineInterval = [np.min(l_point_a[:,1]),np.max(l_point_a[:,1])]

    def poly(x,*coeff):
        return np.poly1d(coeff)(x)    
    
    def generateLinePoint(self,line):
        if (self.polynom is None):
            print("Warming: Polynom wasn't initialized!")
            return 
        l_point_a = LineConvter.TupleList2ArrayList(line)
        l_y = l_point_a[:,1]
        l_x = self.polynom(l_y)
        return LineConvter.XY2TupleList(l_x,l_y)


            
def inWindow(x,y,windowSize):
    return ( (x > 0 and x < windowSize[0]) and (y > 0 and y < windowSize[1]))
        
class LaneMiddleGenerator:
    def __init__(self,laneWidthCm,pxpcm,windowSize):
        self.laneWidthPx =  laneWidthCm * pxpcm
        print("LaneDetector ",self.laneWidthPx)
        self.windowSize = windowSize
        self.nrPoint = 100
        self.pxpcm = pxpcm
        self.stepY = self.windowSize[1] / self.nrPoint
    
    def generateLine (self,polynomlines,middleLine = None):
        line = []
        if(len(polynomlines)>=2):

            middleLineKey = 1.5
            limitMinY = None
            limitMaxY = None
            for key in polynomlines.keys():
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
                            line.append((int(pointX+dX),int(pointY+dY)))
                
                if limitMinY is None or limitMinY > polynomline.lineInterval[0] :
                    limitMinY = polynomline.lineInterval[0]  
                if limitMaxY is None or limitMaxY < polynomline.lineInterval[1] :
                    limitMaxY = polynomline.lineInterval[1]  
            
            if middleLine is None:
                middleLine = PolynomLine(2)
            
            middleLine.estimatePolynom(line)
            newline = []
            for index in range(self.nrPoint):
                pointY = self.stepY * index
                if limitMaxY > pointY and limitMinY < pointY:
                    pointX = middleLine.polynom(pointY)
                    newline.append((int(pointX),int(pointY)))

            middleLine .line = newline   
                
            #     # if (len(pointX_l) > 0):
            #     #     pointX = np.mean(pointX_l)
            #     #     pointY = np.mean(pointY_l)
            #     #     line.append ((int(pointX),int(pointY))) 
            
            return middleLine
        else:
            print(len(polynomlines))
            return None


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
            return [],polynomlines
        print(nonLineKey,maxKey)

        largestLine = polynomlines[maxKey].line
        for key in nonLineKey:
            keyDiff = key - maxKey
            line,polynomlines = self.generatePoint(polynomlines,largestLine,key,keyDiff)
            lines.append(line)
        return lines,polynomlines

    def generatePoint(self,polynomlines,largestLine,key,keyDiff):
        line = []

        # distanceBetweenLine = np.abs(keyDiff) * self.laneWidthPx
        print('KeyDiff:',keyDiff)
        for index in range(len(largestLine)-1):
            pointI = largestLine[index]
            pointI1 = largestLine[index-1]

            dY = pointI1[1] - pointI[1]
            dX = pointI1[0] - pointI[0]
            
            dis = np.sqrt(dY**2 + dX**2)
            rate = self.laneWidthPx*abs(keyDiff) / dis

            if( dY*keyDiff > 0 ):
                dX = -1 * dX
                dY = -1 * dY
            
            newPointX = pointI1[0] - dY * rate
            newPointY = pointI1[1] + dX * rate
            line.append((int(newPointX),int(newPointY)))
        line = sorted(line,key=operator.itemgetter(1))
        polynomlines[key].line = line
        return line,polynomlines




            




        