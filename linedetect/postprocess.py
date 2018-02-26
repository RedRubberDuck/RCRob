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


        
        


class LaneVerifier:
    def __init__(self,laneWidth,pxpcm,errorRate = 0.1):
        self.laneWidthPX = laneWidth*pxpcm
        self.pxpcm = pxpcm
        self.errorLaneWidth = self.laneWidthPX * errorRate
        self.inf_LaneWidth = laneWidth - self.errorLaneWidth
        self.sup_LaneWidth = laneWidth + self.errorLaneWidth
    
    def checkLines(self,lines):
        linePairs = []
        nrLines = len(lines)

        linePairsIndex = [None] * nrLines
        PairsList = [] 
        nrPairs = 0

        print(nrLines)
        if (nrLines <= 1):
            print('Detected nr. lines:',nrLines)
            return lines
        verifiedLines = []
        for i in range(0,nrLines-1):
            
            lineCur = lines[i]
            nrPointCur = len(lineCur)
            for prevI in range(i+1,nrLines):
                linePrev = lines[prevI]
                nrPointPrev = len(linePrev)

                sum_of_error = 0
                # Check the distance between the points on the first Line and the second line
                for j in range(nrPointCur):
                    pointJ = lineCur[j] 
                    minError = None
                    prevError = None
                    for k in range (nrPointPrev):
                        pointK = linePrev[k]
                        distanceJK = distance(pointJ,pointK)
                        error = np.abs(distanceJK - self.laneWidthPX)
                        # if prevError is None:
                        #     prevError  = error
                        # elif prevError * 1.3 < error: 
                        #     # print("Break line Distance")
                        #     break
                        
                        if minError is None or minError > error:
                            minError = error
                    sum_of_error += minError
                sum_of_error /= nrPointCur
                
                print('Sum of error',sum_of_error / self.pxpcm ,' Limit',self.errorLaneWidth / self.pxpcm) 
                if ( sum_of_error < self.errorLaneWidth ):
                    linePairs.append((i,prevI))
                    if(linePairsIndex[i] is None and linePairsIndex[prevI] is None):
                        linePairsIndex[i] = linePairsIndex[prevI] = nrPairs
                        pair = [lines[i],lines[prevI]]
                        PairsList.append(pair)
                        nrPairs += 1
                    elif(linePairsIndex[i] is None):
                        linePairsIndex[i] = linePairsIndex[prevI]
                        PairsList[ linePairsIndex[prevI] ].append(lines[prevI])
                    else:
                        linePairsIndex[prevI] = linePairsIndex[i]
                        PairsList[ linePairsIndex[i] ].append(lines[i])
                    print('Line pair:(',i,',',prevI,')')
        print(linePairsIndex)
        print(PairsList)

        if(len(PairsList)==1):
            return PairsList[0]
        elif(len(PairsList)!=0):
            index = getTheLargestLine(lines)
            pairIndex = linePairsIndex[index]
            return PairsList[pairIndex]
        else:
            index = getTheLargestLine(lines)
            return [lines[index]]
            

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



                    
                    
            

        


        