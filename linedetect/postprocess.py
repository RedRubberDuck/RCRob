import cv2
import numpy as np



def distance(PointI,PointJ):
    return np.sqrt((PointI[0]-PointJ[0])**2 + (PointI[1]-PointJ[1])**2)


## Tt implementing a method with check the lines relation
class Lines:
    def __init__(self,laneWidth,errorRate = 0.05):
        self.laneWidth = laneWidth
        self.errorLaneWidth = laneWidth*errorRate
        self.inf_LaneWidth = laneWidth - self.errorLaneWidth
        self.sup_LaneWidth = laneWidth + self.errorLaneWidth
    
    def checkLines(self,lines):
        nrLines = len(lines)
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
                        error = np.abs(distanceJK - self.laneWidth)
                        # if prevError is None:
                        #     prevError  = error
                        # elif prevError * 1.3 < error: 
                        #     print("Break line Distance")
                        #     break
                        
                        if minError is None or minError > error:
                            minError = error
                    sum_of_error += minError
                sum_of_error /= nrPointCur
                print('I',i,'prevI',prevI,"sum_of_error",sum_of_error,"lane",self.laneWidth)

                    
                    
            

        


        