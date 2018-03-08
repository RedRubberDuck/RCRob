import numpy as np
import cv2
class PointsConnectivity:
    def __init__(self,windowSize):
        self.windowSize = windowSize
        self.maxXDistanceGen = windowSize[0] * 1.0
        # self.maxXDistanceCol = windowSize[0] * 0.5

        self.maxYDistanceGen = windowSize[1] * 3
        # self.maxYDistanceCol = windowSize[1] * 4
        self.limitLine = np.tan(np.radians(15))
        self.limitDistance = np.sqrt(windowSize[0]**2 + windowSize[1]**2)*3.0
        
    def pointColliniarity(self,xDistAB,yDistAB,xDistBC,yDistBC):
        A = yDistBC*xDistAB - xDistBC*yDistAB
        B = (yDistBC*yDistAB + xDistBC*xDistAB)
        B1 = B * self.limitLine
        B2 = B * -self.limitLine
        limitLow = np.min([B1,B2])
        limitHigh = np.max([B1,B2])
        if(limitLow<=A and limitHigh>= A):
            return True
        return False


    def connectPoint(self,windowsCenterAll):
        nrCenter=len(windowsCenterAll)
        line=[]
        # maxXdistance=150
        # maxYDistance=yDistance*4

        listIdsCenter=[None]*nrCenter
        minDistancePoint = [None] * nrCenter
        lines2=[]
        for i in range(nrCenter):
            centerI=windowsCenterAll[i]
            # print(centerI)
            minDistance=None
            minDistanceJ=0
            
            onTheLine = False

            xDistPrev = 0 
            yDistPrev = 0 
            if (listIdsCenter[i] is not None ):
                pointPrev = lines2[listIdsCenter[i]][-1]
                xDistPrev =  pointPrev[0] - centerI[0]
                yDistPrev =  pointPrev[1] - centerI[1]
            
            # Compare with the other points. 
            for j in range(i+1,nrCenter):
                centerJ=windowsCenterAll[j]
                xDist=centerI[0]-centerJ[0]
                yDist=centerI[1]-centerJ[1]
                if(centerI[1]==centerJ[1]):
                    continue
                
                # If the vertical and horizontal distance is smaller than the limits, then we search the nearest point. 
                elif ((abs(xDist)<self.maxXDistanceGen and abs(yDist)<self.maxYDistanceGen and centerI[1]<centerJ[1])) or (listIdsCenter[i] is not None and self.pointColliniarity(xDistPrev,yDistPrev,xDist,yDist)):
                    distance= np.sqrt((xDist)**2 +(yDist)**2)
                    if self.limitDistance>distance and (minDistancePoint[j] is None or minDistancePoint[j] > distance ) and (minDistance is None or  minDistance>distance):
                        minDistance=distance
                        minDistanceJ=j
            # Verifying the nearest point existance, it means the two point can be connected. 
            if minDistance is not None:
                #Verifying the point is a part of the line
                # The examined point is the start point of a new line
                if(listIdsCenter[i] is None):
                    #Getting the new line ID
                    lineId=len(lines2)
                    # Setting the ID in the id and point table
                    listIdsCenter[i]=lineId
                    listIdsCenter[minDistanceJ]=lineId
                    minDistancePoint[minDistanceJ] = minDistance
                    #Creating the new line, as a point container array 
                    line=[centerI]
                    #Adding to the list of the line 
                    lines2.append(line)
                # The examined point is a part of the line
                else:
                    # Getting the line ID
                    lineId=listIdsCenter[i]
                    # Getting the line 
                    line=lines2[lineId]
                    # Adding the new point to the line
                    line.append(centerI)
                    # Setting the line ID for the new point in the table 
                    listIdsCenter[minDistanceJ]=lineId
                    minDistancePoint[minDistanceJ] = minDistance
            # If the point hasn't neighbor point.
            # It takes a part of the line, as a final point 
            elif (listIdsCenter[i] is not None):
                #Final point of a line
                lineId=listIdsCenter[i]
                line=lines2[lineId]
                line.append(centerI)
        return lines2



class LiniarityExaminer:
    def __init__(self,inferiorCorrLimit = 0.9 , lineThinkness = 8):
        self.inferiorCorrLimit = inferiorCorrLimit
        self.lineThinkness = lineThinkness
    def examine(self,frame,verticalTest=True,horizontalTest=True):
        try:
            frame_size=(frame.shape[1],frame.shape[0])
            frame1 = np.array(frame,dtype=np.float)
            rowSum = cv2.reduce(frame1,0,rtype = cv2.REDUCE_SUM)
            try:
                startX = frame_size[0]//2 - rowSum[:frame_size[0]//2][::-1].tolist().index(0)
            except:
                startX = 0
                pass
            try:
                endX = frame_size[0]//2 + rowSum[frame_size[0]//2:].tolist().index(0)
            except:
                endX = frame_size[0]
                pass

            # if startX == endX:
            #     startX = 0
            #     endX = frame_size[0]
            points = cv2.findNonZero(frame[:,startX:endX]) 
            stdDev = cv2.meanStdDev(points)[1]
            points = np.array(points[:,0,:],dtype=np.float)
            mean = None
            res2 = cv2.calcCovarMatrix(points,mean,cv2.COVAR_ROWS +cv2.COVAR_NORMAL)
            covM = res2 [0]
            corrCoef = covM[0,1]/covM[0,0]
            
            point = res2[1][0,0]+startX,res2[1][0,1]
            # Verifying the Pearson correlation coffecients, the vertical line or the horizontal line
            isLinear = (np.abs(corrCoef)>self.inferiorCorrLimit) or (horizontalTest and stdDev[0]<self.lineThinkness*0.341 and stdDev[1]>frame_size[1]*0.22) or  (verticalTest and stdDev[0]>frame_size[0]*0.22 and stdDev[1]<self.lineThinkness*0.341)
            return isLinear,point
        except Exception as e:
            print("Except",e)
            return False,(0,0)
