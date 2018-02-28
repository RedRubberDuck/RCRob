import cv2
import numpy as np



def distance(PointI,PointJ):
    return np.sqrt((PointI[0]-PointJ[0])**2 + (PointI[1]-PointJ[1])**2)

def getTheLargestLine(lines):
        largestLineIndex = None
        largestLineLength = None
        nrLine = len(lines)  
        for i in range(nrLine):
            if largestLineLength is None or largestLineLength < len(lines[i]):
                largestLineLength = len(lines[i])
                largestLineIndex = i
        return largestLineIndex


## It implementing a method with check the lines relation
class LaneLineEstimator:
    def __init__(self,laneWidth,pxpcm):
        self.laneWidthPx = laneWidth * pxpcm
        self.pxpcm = pxpcm
    

    def estimateLine(self,lines):
        Index = self.getTheLargestLine(lines)
        # Index = 0

        # newLines = []
        L1 = []
        L2 = []
        L3 = []
        L4 = []
        curLine = lines[Index]
        nrLine = len(curLine)

        for i in range(0,nrLine-2):
            PointI = curLine[i]
            PointI1 = curLine[i+1]
            PointI2 = curLine[i+2]

            dX1 = PointI1[0] - PointI[0]
            dY1 = PointI1[1] - PointI[1]

            dX2 = PointI2[0] - PointI1[0]
            dY2 = PointI2[1] - PointI1[1]

            dX = (dX1 + dX2) / 2
            dY = (dY1 + dY2) / 2



            dist = np.sqrt(dX **2 + dY **2 )
            dist1 = np.sqrt(dX1 **2 + dY1 **2 )
            
            rate = self.laneWidthPx / dist  
            rate1 = self.laneWidthPx / dist1  

            newPoint1X = int(PointI1[0] + dY*rate)
            newPoint1Y = int(PointI1[1] - dX*rate)

            newPoint2X = int(PointI1[0] - dY*rate)
            newPoint2Y = int(PointI1[1] + dX*rate)
            
            newPoint1X_p = int(PointI1[0] + dY1*rate1)
            newPoint1Y_p = int(PointI1[1] - dX1*rate1)

            newPoint2X_p = int(PointI1[0] - dY1*rate1)
            newPoint2Y_p = int(PointI1[1] + dX1*rate1)

            L1.append((newPoint1X,newPoint1Y))
            L2.append((newPoint2X,newPoint2Y))
            
            L3.append((newPoint1X_p,newPoint1Y_p))
            L4.append((newPoint2X_p,newPoint2Y_p))

        return [L1,L2,L3,L4]


        
        


class LaneVerifierBasedDistance:
    def __init__(self,laneWidth,pxpcm,errorCm =5):
        self.laneWidthPX = laneWidth*pxpcm
        self.pxpcm = pxpcm
        self.errorLaneWidth = errorCm * pxpcm 
        self.inf_LaneWidth = laneWidth - self.errorLaneWidth
        self.sup_LaneWidth = laneWidth + self.errorLaneWidth
    

    def checkDistanceBetweenMinLines(self,lineI,lineJ):
        nrPointLineI = len(lineI)
        nrPointLineJ = len(lineJ)
        if nrPointLineI <= nrPointLineJ:
            refLine = lineI
            exLine  = lineJ
        else:
            refLine = lineJ
            exLine  = lineI
        
        sum_of_dis = 0
        for i in range(len(refLine)):
            pointRef = refLine[i]
            minDistance = None
            for j in range(len(exLine)):
                pointEx = exLine[j]
                dis = distance(pointRef,pointEx)
                if minDistance is None or minDistance > dis:
                    minDistance = dis
            sum_of_dis += minDistance
        meanDistance = sum_of_dis/ len(refLine)
        print(np.abs(self.laneWidthPX - meanDistance))        
        return (np.abs(self.laneWidthPX - meanDistance) < self.errorLaneWidth)


    def checkLane(self,lines):
        nrLines = len(lines)
        laneIndexforLine = [None] * nrLines
        LaneList = [] 
        nrLane = 0

        if (nrLines <= 1):
            print('Detected nr. lines:',nrLines)
            return lines
        verifiedLines = []
        for curI in range(0,nrLines-1):
            lineCur = lines[curI]
            for prevI in range(curI+1,nrLines):
                linePrev = lines[prevI]
                isLane = self.checkDistanceBetweenMinLines(lineCur,linePrev)
                if isLane:
                    if (laneIndexforLine[curI] is None and laneIndexforLine[prevI] is None):
                        laneIndexforLine[curI] = nrLane
                        laneIndexforLine[prevI] = nrLane
                        LaneList.append([curI,prevI])
                        nrLane += 1
                    elif(laneIndexforLine[curI] is None):
                        laneIndexforLine[curI] = laneIndexforLine[prevI]
                        LaneList[laneIndexforLine[prevI]].append(curI)
                    elif(laneIndexforLine[prevI] is None):
                        laneIndexforLine[prevI] = laneIndexforLine[curI]
                        LaneList[laneIndexforLine[curI]].append(prevI)

        if (nrLane!=1):
            print("Lane not found! Please check the input Image.")
        elif len(LaneList[0])>3:
            print("To much line detected for a lane! Please check the input Image.")
        else:
             newlines = []
             for i in LaneList[0]:
                 newlines.append(lines[i])
        return newlines



def TupleLineToComplexLine(line):
    complexLine = []
    for point in line:
        complexLine.append(complex(point[0],point[1]))
def TupleLineToArrayLine(line):
    complexLine = []
    for point in line:
        complexLine.append([point[0],point[1]])

def XYToTupleList(X,Y)
    
    line = []
    for x,y in zip(X,Y):
        line.append((int(x),int(y)))
    return line

class LinePolynom:
    def __init__(self,polynomDeg):
        self.polynomDeg = polynomDeg
        self.poly = None
    def estimate(self,line):
        if len(line) < self.polynomOrder:
            print("Not enough point!")
            return 
        point_a = np.array(TupleLineToArrayLine(line))
        X = point_a[:,0]
        Y = point_a[:,1]
        self.poly = np.polyfit(Y,X,deg=self.polynomDeg)
    
    def reference(self,lines):
        if self.poly == None:
            print("The polynom  wasn't initialized!")
            return 
        point_a = np.array(TupleLineToArrayLine(line))
        Y = point_a[:,1]
        X = self.poly(Y)
        return XYToTupleList(X,Y)

        


    

                    
                    
            

        


        