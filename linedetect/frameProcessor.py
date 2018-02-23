import cv2
import numpy as np
import frameFilter

import cv2.plot
from matplotlib import pyplot as plt

## Implementing the perspective transformation functionality
class ImagePersTrans:
    ## Constructor
    #   @param: M                   The transformation matrix to the new perspective view
    #   @param: M_inv               The matrix of the iverse perspective transformation 
    #   @param: size                The size of the new frame after transformation
    def __init__(self,M,M_inv,size):
        self.M = M
        self.M_inv = M_inv
        self.size = size
    def wrapPerspective(self,frame):
        return cv2.warpPerspective(frame,self.M,self.size)

    # Getting the transformation matrix, the invers transformation matrix, the size of the transformated image for the first camera position
    def getPerspectiveTransformation():
        corners_pics = np.float32([
                [434-70,527],[1316+70,497],
                [-439,899],[2532,818]])

        step=45*5
        corners_real = np.float32( [
                [0,0],[2,0],
                [0,2],[2,2]])*step

        M = cv2.getPerspectiveTransform(corners_pics,corners_real)
        M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
        return ImagePersTrans(M,M_inv,(int(step*2),int(step*2)))


class SlidingWindowMethod:
    def __init__(self,nrSlice,windowSize):
        self.nrSlice = nrSlice
        self.windowSize = windowSize 
        self.histogramProc = HistogramProcess(0.002777778,0.023570226,lineThinkness =  12*2,xDistanceLimit = windowSize[0]//2)
        self.liniarityExaminer = LiniarityExaminer(inferiorCorrLimit = 0.9, lineThinkness = 12*2)
        self.pointConnectivity = PointsConnectivity(windowSize)
    def apply(self,img_bin):
        img_size=(img_bin.shape[1],img_bin.shape[0])
        windowsCenterAll=[]
        for i in range(self.nrSlice):
            part=img_bin[int(img_size[1]*i/self.nrSlice):int(img_size[1]*(i+1)/self.nrSlice),:]
            yPos=int(img_size[1]*(i+0.5)/self.nrSlice)
            windowsCenter=self.histogramProc.histogramMethod(part,yPos)
            windowsCenterAll+=windowsCenter
        windowsCenterAll = self.pointsLineVerifying(img_bin,windowsCenterAll)
        lines = self.pointConnectivity.connectPoint(windowsCenterAll)
        return windowsCenterAll,lines

    def pointsLineVerifying(self,frame,points):
        points_ = []
        frameSize = (frame.shape[1],frame.shape[0])
        for point in points:
            # Window index
            limitsY=np.array([point[1]-self.windowSize[1]//2,point[1]+self.windowSize[1]//2],dtype=np.int32)
            limitsX=np.array([point[0]-self.windowSize[0]//2,point[0]+self.windowSize[0]//2],dtype=np.int32)
            # Clipping 
            limitsX=np.clip(limitsX,0,frameSize[0])
            limitsY=np.clip(limitsY,0,frameSize[1])
            # 
            part=frame[limitsY[0]:limitsY[1],limitsX[0]:limitsX[1]]

            isline=self.liniarityExaminer.examine(part,verticalTest = False)

            if isline:
                points_.append(point)

        return points_
        

class PointsConnectivity:
    def __init__(self,windowSize):
        self.windowSize = windowSize
        self.maxXDistanceGen = windowSize[0] * 1.0
        # self.maxXDistanceCol = windowSize[0] * 0.5

        self.maxYDistanceGen = windowSize[1] * 3
        # self.maxYDistanceCol = windowSize[1] * 4
        self.limitLine = np.tan(np.radians(15))
        self.limitDistance = 450
        
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
            rowSum = np.sum(frame,axis=0)
            # print(len(rowSum),frame_size)
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
            # print(startX,endX) 

            if startX == endX:
                startX = 0
                endX = frame_size[0]

            # cv2.imshow('ss',frame[:,startX:endX])
            # cv2.waitKey()
            # Getting the non-zero coordinates in the window
            Y,X=np.nonzero(frame[:,startX:endX])
            # Verifying the number of non-zero points
            if(X.shape[0]<3):
                # print("S:",X.shape)
                return False
            # Standard deviation calculating
            StdX=np.std(X)
            StdY=np.std(Y)
            # Pearson/Spearman correlation coefficients 
            corrCoef=np.corrcoef(Y,X)[0,1]
            # Verifying the Pearson correlation coffecients, the vertical line or the horizontal line
            return (np.abs(corrCoef)>self.inferiorCorrLimit) or (horizontalTest and StdX<self.lineThinkness*0.341 and StdY>frame_size[1]*0.22) or  (verticalTest and StdX>frame_size[0]*0.22 and StdY<self.lineThinkness*0.341)
        except Exception as e:
            print("Except",e)
            return False



class HistogramProcess:
    def __init__(self,inferiorRate,superiorRate,lineThinkness,xDistanceLimit):
        self.superiorRate = superiorRate
        self.inferiorRate = inferiorRate
        self.xDistanceLimit = xDistanceLimit
        # self.lineThinkness = lineThinkness
        self.kernel =  (np.ones((1,lineThinkness))/lineThinkness)[0,:]

    def histogramMethod(self,part,yPos):
        windowscenter=[]
        part_size=(part.shape[1],part.shape[0])
        #Limit calculation
        slice_size=part_size[1]*part_size[0]
        
        inferiorLimitSize = slice_size*self.inferiorRate
        superiorLimitSize = slice_size*self.superiorRate

        #Calculating histogram
        histogram=np.sum(part,axis=0)/255

        #Filter the histogram
        histogram_f=np.convolve(histogram,self.kernel,'same')
        # histogram_f = histogram

        accumulate=0
        accumulatePos=0
        accumulate_a=[]
        for i in range(part_size[0]):
            #The non-zero block
            if histogram_f[i]>0:
                accumulate+=histogram_f[i]
                accumulate_a.append(accumulate)
                accumulatePos+=histogram_f[i]*i
                
            # The end of a non-zero block
            elif histogram_f[i]==0 and histogram_f[i-1]>0:
                
                if accumulate<superiorLimitSize and accumulate>inferiorLimitSize:
                    #Calculating the middlsuperiorLimitSizee of the non-zero block
                    indexP=int(accumulatePos/accumulate)
                    #Verify the distance from the last non-zeros block
                    if (len(windowscenter)>0 and abs(windowscenter[-1][0]-indexP)<self.xDistanceLimit):
                        #If the distance is smaller than the threshold, then combines it.
                        indexP=int((windowscenter[-1][0]+indexP)/2)
                        windowscenter[-1]=(indexP,windowscenter[-1][1])
                    else:
                        # Add to the list of the windowsCenters
                        windowscenter.append((indexP,yPos))
                accumulate=0
                accumulatePos=0
            accumulate_a.append(accumulate)
                

        plotObj = cv2.plot.Plot2d_create(-histogram)
        plotObj.setPlotSize(part_size[0],int(np.max(histogram)))
        return windowscenter


def generatingNewPosition(lines,windowYSize):
    for line in lines:
        nrPoint = len(line)
        nrNewPoint=0
        for i in range(nrPoint-1):
            centerI = line[i+nrNewPoint]
            centerI1 = line[i+1+nrNewPoint]
            disY = centerI1[1] - centerI[1]
            nrLineDis=int((disY)/windowYSize)
            # print(nrLineDis)
            for j in range(1,nrLineDis):
                posX = int(centerI[0] + ((centerI1[0]-centerI[0])*j/nrLineDis ))
                posY = int(centerI[1] + ((centerI1[1]-centerI[1])*j/nrLineDis ))
                line.insert(i+j+nrNewPoint,(posX,posY))
            if nrLineDis>1:
                nrNewPoint+=(nrLineDis-1)
            # Generating new point
            # for j in range(1,nrLineDis):
    # print(len(lines))




class NonSlidingWindowMethod:

    def windowCutting(im,pos,windowSize):
        img_size=(im.shape[1],im.shape[0])
        startX=int(pos[0]-windowSize[0]/2)
        endX=int(pos[0]+windowSize[0]/2)
        
        startY=int(pos[1]-windowSize[1]/2)
        endY=int(pos[1]+windowSize[1]/2)

        [startX,endX]=np.clip([startX,endX],0,img_size[0])
        [startY,endY]=np.clip([startY,endY],0,img_size[1])
        window=im[startY:endY,startX:endX]
        return window,startX
        
    def generatingNewPosition(lines,windowYSize):
        for line in lines:
            nrPoint = len(line)
            nrNewPoint=0
            for i in range(nrPoint-1):
                centerI = line[i+nrNewPoint]
                centerI1 = line[i+1+nrNewPoint]
                disY = centerI1[1] - centerI[1]
                nrLineDis=int((disY)/windowYSize)
                # print(nrLineDis)
                for j in range(1,nrLineDis):
                    posX = int(centerI[0] + ((centerI1[0]-centerI[0])*j/nrLineDis ))
                    posY = int(centerI[1] + ((centerI1[1]-centerI[1])*j/nrLineDis ))
                    line.insert(i+j+nrNewPoint,(posX,posY))
                if nrLineDis>1:
                    nrNewPoint+=(nrLineDis-1)
                # Generating new point
                # for j in range(1,nrLineDis):
        # print(len(lines))
    
    def nonslidingWindowMethod(mask,linesCenterPos,windowSize):
        lineEximiner = LiniarityExaminer(inferiorCorrLimit = 0.9 ,lineThinkness = 24)

        img_size=(mask.shape[1],mask.shape[0])
        #Preprocessing each line to add new points, when the first point of the line is not in the first slice and the last point of the line is not in the last slice 
        NonSlidingWindowMethod.generatingNewPosition(linesCenterPos,windowSize[1])
        
        nrSliceInImg=int(img_size[1]/windowSize[1])
        for line in linesCenterPos:
            firstPoint=line[0]
            lineFirstPoint=int(firstPoint[1]/windowSize[1])
            if (lineFirstPoint>0):
                newPoint=(firstPoint[0],firstPoint[1]-windowSize[1])
                line.insert(0,newPoint)
            lastPoint=line[-1]
            lineLastPoint=int(firstPoint[1]/windowSize[1])
            if (lineLastPoint<nrSliceInImg-1):
                newPoint=(lastPoint[0],lastPoint[1]+windowSize[1])
                line.append(newPoint)
            
        
        # Processing each lines
        for line in linesCenterPos:
            nrPoint=len(line)
            print('Innit',nrPoint)
            nrRemovedPoint=0
            for i in range(nrPoint):
                pos=line[i-nrRemovedPoint]
                
                
                window,startX=NonSlidingWindowMethod.windowCutting(mask,pos,windowSize)
                sumWhiteX=np.sum(window,axis=0)/255
                sumTotalWhite=np.sum(sumWhiteX)
                print('SumTotal White',sumTotalWhite)
                if sumTotalWhite>30:
                    posX=np.average(range(len(sumWhiteX)),weights=sumWhiteX)   
                    # windowT,s= NonSlidingWindowMethod.windowCutting(mask,pos,(windowSize[0],windowSize[1]))
                    # print(windowT.shape)
                    isLine = lineEximiner.examine(window)
                    # isLine = True
                    if isLine:
                        posNew=(int(startX+posX),pos[1])
                        line[i-nrRemovedPoint]=posNew
                    else:
                        # ss=0
                        nrRemovedPoint+=1
                        print(len(line))
                        line.remove(pos)
                        print('Fin',len(line))
                else:
                    s=0
                    nrRemovedPoint+=1
                    print(len(line))
                    line.remove(pos)
                    print('Fin',len(line))
            #  Verify number of point in the line 
            if len(line) == 0:
                # The line disappeared
                linesCenterPos.remove(line)
            print('Int',len(line))
        print(len(linesCenterPos))
        return linesCenterPos