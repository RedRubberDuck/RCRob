import numpy as np
import cv2

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
                dist  = abs(lineRef[i]-lineEx[j])

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
            # print('Detected nr. lines:',nrLines)
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
            # print("Line",Lanes[0])
            newLines = []
            for index in Lanes[0]:
                newLines.append(lines[index])
            return newLines
        else:
            print("Distance based lane grouping failedd!",nrLane)
        return []
