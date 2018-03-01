import cv2
import numpy as np
import operator



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
        return np.abs(mean_of_distance - self.laneWidthPX) < self.errorLaneWidthPX

    
    def checkLane(self,lines):
        nrLines = len(lines)

        laneIndex = [None] * nrLines
        Lanes = [] 
        nrLane = 0

        print(nrLines)
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
            newLines = []
            for index in Lanes[0]:
                newLines.append(lines[index])
            return newLines
        else:
            print("Distance based lane grouping failedd!",nrLane)
        return None 
            

    # def checkThePairs(self,lines,linePairsIndex):
    #     nrPairs = np.max(linePairsIndex)
    #     resLine = []
    #     if(nrPairs == 1):
    #         nrLines = len(lines)
    #         nrRemoved = 0
    #         for i in range(nrLines):
    #             if(linePairsIndex[i] is not None):
    #                 resLine.append(lines[i])
    #     else:



                    



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
        X = polynomLine.polynom(Y)
        lineTestPos_dic[key]=X
    sorted_line = sorted(lineTestPos_dic.items(), key=operator.itemgetter(1))

    newPolynomLine_dic = {}
    for index in range(len(sorted_line)):
        key,pointX = sorted_line[index]
        print(index,key)
        newPolynomLine_dic[index] = polynomLine_dic[key]
    
    return newPolynomLine_dic



class PolynomLine:
    def __init__(self,polyDeg):
        self.polyDeg = polyDeg
        self.polynom = None
        self.dPolynom = None 
        self.line = None
    def estimatePolynom(self,line):
        if len(line) <= self.polyDeg:
            print("Warming: Not enough to estimate the polynom")
            return
        l_point_a = np.array(line)
        l_y = l_point_a[:,1]
        l_x = l_point_a[:,0]
        coeff = np.polyfit(l_y,l_x,self.polyDeg) 
        self.polynom = np.poly1d(coeff)
        self.dPolynom = self.polynom.deriv()
        self.line = line
    
    def generateLinePoint(self,line):
        if (self.polynom is None):
            print("Warming: Polynom wasn't initialized!")
            return 
        l_point_a = LineConvter.TupleList2ArrayList(line)
        l_y = l_point_a[:,1]
        l_x = self.polynom(l_y)
        return LineConvter.XY2TupleList(l_x,l_y)


            

        


        