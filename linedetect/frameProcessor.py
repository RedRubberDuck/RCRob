import cv2
import numpy as np
import frameFilter, postprocess, PointProcess
import cv2.plot
from matplotlib import pyplot as plt


class SlidingWindowMethod:
    def __init__(self,nrSlice,frameSize,windowSize):
        self.nrSlice = nrSlice
        self.windowSize = windowSize 

        partSize = (frameSize[0], frameSize[1]//nrSlice)
        print('Window size:',windowSize,'Part size:',partSize)
        self.histogramProc = HistogramProcess(0.002777778,0.023570226,lineThinkness =  2*5,xDistanceLimit = windowSize[0]//2,partSize=partSize)
        self.liniarityExaminer = PointProcess.LiniarityExaminer(inferiorCorrLimit = 0.9, lineThinkness = 2*5)
        self.pointConnectivity = PointProcess.PointsConnectivity(windowSize)
    def apply(self,img_bin):
        img_size=(img_bin.shape[1],img_bin.shape[0])
        windowsCenterAll=[]
        for i in range(self.nrSlice):
            part=img_bin[int(img_size[1]*i/self.nrSlice):int(img_size[1]*(i+1)/self.nrSlice),:]
            yPos=int(img_size[1]*(i+0.5)/self.nrSlice)
            windowsCenter=self.histogramProc.histogramMethod(part,yPos)
            windowsCenterAll+=windowsCenter
        # windowsCenterAll = self.pointsLineVerifying(img_bin,windowsCenterAll)
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

            isline,point=self.liniarityExaminer.examine(part,verticalTest = False)

            if isline:
                points_.append(point)

        return points_
        







class HistogramProcess:
    def __init__(self,inferiorRate,superiorRate,lineThinkness,xDistanceLimit,partSize):
        self.superiorRate = superiorRate
        self.inferiorRate = inferiorRate
        self.xDistanceLimit = xDistanceLimit
        # self.lineThinkness = lineThinkness
        self.kernel =  (np.ones((1,lineThinkness))/lineThinkness)[0,:]
        self.partSize  = partSize
        partArea = partSize[1]*partSize[0]

        self.inferiorLimitSize = partArea * self.inferiorRate
        self.superiorLimitSize = partArea * self.superiorRate

    def histogramMethod(self,part,yPos):
        windowscenter=[]

        #Calculating histogram
        part1 = np.array(part,dtype=np.float)
        histogram = cv2.reduce(part1,0,rtype = cv2.REDUCE_SUM)[0,:]/255.0
        #Filter the histogram
        
        histogram_f=np.convolve(histogram,self.kernel,'same')
        # histogram_f = histogram

        accumulate=0
        accumulatePos=0
        startPx = 0
        # accumulate_a=[]
        for i in range(1,self.partSize[0]):
            #The non-zero block
            if histogram_f[i]>0:
                if histogram_f[i-1]==0:
                    startPx = i
                accumulate += histogram_f[i]
                # accumulatePos += histogram_f[i]*i
            # The end of a non-zero block
            elif histogram_f[i]==0 and histogram_f[i-1]>0:
                
                if accumulate<self.superiorLimitSize and accumulate> self.inferiorLimitSize:
                    #Calculating the middlsuperiorLimitSizee of the non-zero block
                    indexP=int((startPx+i)/2)
                    #Verify the distance from the last non-zeros block
                    if (len(windowscenter)>0 and abs(windowscenter[-1][0]-indexP)<self.xDistanceLimit):
                        #If the distance is smaller than the threshold, then combines it.
                        indexP=int((windowscenter[-1][0]+indexP)/2)
                        windowscenter[-1]=(indexP,windowscenter[-1][1])
                    else:
                        # Add to the list of the windowsCenters
                        windowscenter.append((indexP,yPos))
                accumulate=0
                # accumulatePos=0
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


class NonSlidingWindowMethodWithPolynom:
    def __init__(self,windowSize,distanceLimit,lineThinknessPx):
        self.windowSize = windowSize
        self.lineEximiner = PointProcess.LiniarityExaminer(inferiorCorrLimit = 0.75 ,lineThinkness = lineThinknessPx)
        
        self.distanceLimit = distanceLimit
        self.supLimitNrNonZero = np.max(windowSize) * lineThinknessPx * 1.6
        self.infLimitNrNonZero = np.min(windowSize) * lineThinknessPx * 0.8

    def windowCutting(im,pos,windowSize):
        img_size=(im.shape[1],im.shape[0])
        startX=int(pos[0]-windowSize[0]/2)
        endX=int(pos[0]+windowSize[0]/2)
        
        startY=int(pos[1]-windowSize[1]/2)
        endY=int(pos[1]+windowSize[1]/2)

        [startX,endX]=np.clip([startX,endX],0,img_size[0])
        [startY,endY]=np.clip([startY,endY],0,img_size[1])
        window=im[startY:endY,startX:endX]
        return window,startX,startY
        

    def simplifyLine(self,line):
        nrPoint =  len(line)
        nrRemoved = 0
        for i in range(nrPoint-1):
            pointI = line[i-nrRemoved]
            pointJ = line[i + 1 -nrRemoved]
            disX = pointJ[0] - pointI[0]
            disY = pointJ[1] - pointI[1]
            dis = np.sqrt(disX**2 + disY**2)
            if dis < self.distanceLimit:
                newPointX = (pointI[0] + pointJ[0])//2
                newPointY = (pointI[1] + pointJ[1])//2
                newPoint = (newPointX,newPointY)
                line[i - nrRemoved] = newPoint
                line.remove(pointJ)
                nrRemoved += 1
    
    def addFrontPoint(self,imageSize,polynomLine,nrNewPoint=3):
        frontPoint = polynomLine.line[0]
        for i in range(nrNewPoint):
            dV = polynomLine.dPolynom(frontPoint[1])
            dY = self.distanceLimit*1.0  /np.sqrt(dV**2+1)
            newPointY = int(frontPoint[1]-dY)
            newPointX = int(polynomLine.polynom(newPointY))
            
            if (newPointX > 0 and newPointX < imageSize[0]) and (newPointY > 0 and newPointY < imageSize[1]):
                polynomLine.line.insert(0,(newPointX,newPointY))
                frontPoint = (newPointX,newPointY)
            else:
                break
    
    def addBackPoint(self,imageSize,polynomLine,nrNewPoint=3):
        backPoint = polynomLine.line[-1]
        # print(backPoint)
        for i in range(nrNewPoint):
            dV = polynomLine.dPolynom(backPoint[1])
            
            dY = self.distanceLimit*1.0  /np.sqrt(dV**2+1)
            # print(dY,dY*dV)
            newPointY = int(backPoint[1]+dY)
            newPointX = int(polynomLine.polynom(newPointY))
            
            if (newPointX > 0 and newPointX < imageSize[0]) and (newPointY > 0 and newPointY < imageSize[1]):
                polynomLine.line.append((newPointX,newPointY))
                backPoint = (newPointX,newPointY)
                # print(backPoint)
            else:
                break
    
    def addIntermediatPoint(self,polynomLine):
        line = polynomLine.line
        nrPoint = len(line)
        nrNewPoint=0
        for i in range(nrPoint-1):
            pointI = line[i+nrNewPoint]
            pointI1 = line[i+1+nrNewPoint]
            
            disY = pointI1[1] - pointI[1]
            disX = pointI1[0] - pointI[0]
            dist =  np.sqrt(disY**2 + disX**2)
            
            nrLineDis=int(dist/self.distanceLimit/1.1)
            # print("NR.Lines:",nrLineDis)
            for j in range(1,nrLineDis):
                newPointY = int(pointI[1] + (disY*j/nrLineDis ))
                newPointX = int(polynomLine.polynom(newPointY))
                line.insert(i+j+nrNewPoint,(newPointX,newPointY))
            if nrLineDis>1:
                nrNewPoint+=(nrLineDis-1)
    

    def checkPoint(self,point,image):
        window,startX,startY=NonSlidingWindowMethod.windowCutting(image,point,self.windowSize)
        histWhiteX = np.sum(window,axis=0)/255
        histWhiteY = np.sum(window,axis=1)/255        
        
    
    def lineProcess(self,mask,polynomLine):
        line = polynomLine.line
        if len(line) == 0:
            return 
        
        if polynomLine.polynom is not None:
            self.addFrontPoint(mask.shape,polynomLine)
            self.addBackPoint(mask.shape,polynomLine)
            self.addIntermediatPoint(polynomLine)

        nrPoint = len(line)
        nrRemovedPoint = 0
        # Check all point 
        for index in range(nrPoint):
            point = line[index - nrRemovedPoint]
            # Copy the surrounding area of the point
            window,startX,startY=NonSlidingWindowMethodWithPolynom.windowCutting(mask,point,self.windowSize)


            nrNonZero = cv2.countNonZero(window)
            # print(nrNonZero
            if nrNonZero > self.infLimitNrNonZero and nrNonZero < self.supLimitNrNonZero:
                isLine,pointPlus = self.lineEximiner.examine(window)
                if isLine :
                    
                    pointNew=(int(startX+pointPlus[0]),int(startY+pointPlus[1]))
                    line[index-nrRemovedPoint]=pointNew
                else:
                    nrRemovedPoint+=1
                    line.remove(point)
            else:
                # print(nrNonZero, self.infLimitNrNonZero,self.supLimitNrNonZero)
                nrRemovedPoint+=1
                line.remove(point)
        self.simplifyLine(line)
        if len(line) < 3:
            polynomLine.line = []
            return
        polynomLine.estimatePolynom(line)
        # Check the length of the line


    def nonslidingWindowMethod(self,mask,polynomline_dic):
        img_size=(mask.shape[1],mask.shape[0])
        for polynomline_Key in polynomline_dic:
            self.lineProcess(mask,polynomline_dic[polynomline_Key]) 
        return