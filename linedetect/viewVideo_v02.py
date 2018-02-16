import cv2, time, os
from matplotlib import pyplot as plt
import numpy as np

x=np.linspace(0,2,20)
sig=0.4;mu=1
gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
kernel=gauss/np.sum(gauss)

def drawLine(gray,lines):
    nrLines=len(lines)
    for i in range(0,nrLines-1):
        startpoint=lines[i]
        endPoint=lines[i+1]
        cv2.line(gray,startpoint,endPoint,thickness=1,color=(255-nrLines,255,255))
    return gray


def drawWindows(gray,windowsCenter,windowSize):

    for center in windowsCenter:
        points=np.array([[[center[0]-windowSize[0]/2,center[1]-windowSize[1]/2],
                            [center[0]+windowSize[0]/2,center[1]-windowSize[1]/2],
                            [center[0]+windowSize[0]/2,center[1]+windowSize[1]/2],
                            [center[0]-windowSize[0]/2,center[1]+windowSize[1]/2]]],dtype=np.int32)
        gray=cv2.polylines(gray,points,thickness=1,isClosed=True,color=(255,255,255))
    return gray


def slidingWindowMethod(gray,mask, nrSlices):
    #filter bog and small objects
    
    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # print(nlabels)
    
    # for labeli in range(1,nlabels)

    img_size=(mask.shape[1],mask.shape[0])
    windowsCenterAll=[]
    windowsCenterSlice=[]
    for i in range(nrSlices):
        part=mask[int(img_size[1]*i/nrSlices):int(img_size[1]*(i+1)/nrSlices),:]
        yPos=int(img_size[1]*(i+0.5)/nrSlices)
        windowsCenter=histogramMethod2(part,yPos)
        windowsCenterAll+=windowsCenter
        windowsCenterSlice.append(windowsCenter)
    lines=postProcessWindowCenter(windowsCenterSlice,windowsCenterAll,img_size[1]/nrSlices)
    for line in lines:
        gray=drawLine(gray,line)
    # gray=drawLine(gray,lines[2])
    gray=drawWindows(gray,windowsCenterAll,(int(mask.shape[1]*3/15),int(mask.shape[0]/15)))
    return gray,lines

def postProcessWindowCenter(windowsCenter,windowsCenterAll,yDistance):
    nrCenter=len(windowsCenterAll)
    line=[]
    maxXdistance=150
    maxYDistance=yDistance*3

    listIdsCenter=[None]*nrCenter
    lines2=[]
    for i in range(nrCenter):
        centerI=windowsCenterAll[i]
        minDistance=None
        minDistanceJ=0
        # Compare with the other points. 
        for j in range(i,nrCenter):
            centerJ=windowsCenterAll[j]
            # If the two point is the same line, they cannot be connected
            if(centerI[1]==centerJ[1]):
                continue
            # If the vertical and horizontal distance is smaller than the limits, then we search the nearest point. 
            elif abs(centerI[0]-centerJ[0])<maxXdistance and abs(centerI[1]-centerJ[1])<maxYDistance and centerI[1]<centerJ[1]:
                distance= np.sqrt((centerI[0]-centerJ[0])**2 +(centerI[1]-centerJ[1])**2)
                if minDistance is None or  minDistance>distance:
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
        # If the point hasn't neighbor point.
        # It takes a part of the line, as a final point 
        elif (listIdsCenter[i] is not None):
            #Final point of a line
            lineId=listIdsCenter[i]
            line=lines2[lineId]
            line.append(centerI)
        
    return lines2


def histogramMethod2(part,yPos):
    windowscenter=[]
    part_size=(part.shape[1],part.shape[0])
    #Limit calculation
    slice_size=part_size[1]*part_size[0]
    # upper_limit=0.037037037037037035 
    upper_limit=0.04537037037037035 
    lower_limit=0.009259259259259259
    upperLimitSize = slice_size*upper_limit
    loweLimitSize = slice_size*lower_limit

    #Calculating histogram
    histogram=np.sum(part,axis=0)/255
    #Filter the histogram
    histogram_f=np.convolve(histogram,kernel,'same')

    accumulate=0
    accumulatePos=0
    for i in range(part_size[0]):
        #The non-zero block
        if histogram_f[i]>0:
            accumulate+=histogram_f[i]
            accumulatePos+=histogram_f[i]*i
        # The end of a non-zero block
        elif histogram_f[i]==0 and histogram_f[i-1]>0:
            if accumulate<upperLimitSize and accumulate>loweLimitSize:
                #Calculating the middle of the non-zero block
                indexP=int(accumulatePos/accumulate)
                # print(indexP,' ',indexAA)
                #Verify the distance from the last non-zeros block
                if (len(windowscenter)>0 and abs(windowscenter[-1][0]-indexP)<100):
                    #If the distance is smaller than the threshold, then combines it.
                    indexP=int((windowscenter[-1][0]+indexP)/2)
                    windowscenter[-1]=(indexP,windowscenter[-1][1])
                else:
                    # Add to the list of the windowsCenters
                    windowscenter.append((indexP,yPos))
            accumulate=0
            accumulatePos=0
    
    return windowscenter

def windowClip(im,pos,windowSize):
    img_size=(im.shape[1],im.shape[0])
    startX=int(pos[0]-windowSize[0]/2)
    endX=int(pos[0]+windowSize[0]/2)
    if(startX<0):
        startX=0
    if(endX>=img_size[0]):
        endX=img_size[0]-1

    startY=int(pos[1]-windowSize[1]/2)
    endY=int(pos[1]+windowSize[1]/2)
    if(startY < 0 ):
        startY=0
    if(endY >=img_size[1]):
        endY=img_size[1]-1
    window=im[startY:endY,startX:endX]
    return window,startX

def nonslidingWindowMethod(mask,linesCenterPos,windowSize):
    img_size=(mask.shape[1],mask.shape[0])
    #Preprocessing each line to add new points, when the first point of the line is not in the first slice and the last point of the line is not in the last slice 
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
        nrRemovedPoint=0
        for i in range(nrPoint):
            pos=line[i-nrRemovedPoint]
            
            window,startX=windowClip(mask,pos,windowSize)
            sumWhiteX=np.sum(window,axis=0)
            sumTotalWhite=np.sum(sumWhiteX)
            if sumTotalWhite>0:
                posX=np.average(range(len(sumWhiteX)),weights=sumWhiteX)    
                pos=(int(startX+posX),pos[1])
                line[i-nrRemovedPoint]=pos
            else:
                s=0
                nrRemovedPoint+=1
                line.remove(pos)
        #  Verify number of point in the line 
        if len(line) == 0:
            # The line disappeared
            linesCenterPos.remove(line)

    return linesCenterPos
                
def videoRead(fileName):
    cap = cv2.VideoCapture()
    cap.open(fileName)
    # print('OK')
    index=0

    durationTime=int(1.0/30.0*1000.0)

    while (cap.isOpened()):
        ret,frame = cap.read()
        if(ret):
            # cv2.imshow('',frame)
            # cv2.waitKey(durationTime)
            # time.sleep(durationTime/1000.0)
            yield frame,durationTime
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



def processFrame2(frame):
    img_size=(frame.shape[1],frame.shape[0])

    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    hsv[:,:,1] = clahe.apply(hsv[:,:,1])
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])

    lower_white = np.array([0,0,140], dtype=np.uint8)
    upper_white = np.array([255,80,255], dtype=np.uint8)
    # #---------------------------------------------------------------------------

    mask = cv2.inRange(hsv, lower_white, upper_white)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    return gray,mask

def processFrame5(frame):
    img_size=(frame.shape[1],frame.shape[0])
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(7,7))

    lower_gray_bgr = np.uint8([[[118,118,118]]])
    lower_gray_ = cv2.cvtColor(lower_gray_bgr,cv2.COLOR_BGR2YCrCb)

    high_gray_bgr = np.uint8([[[255,255,255]]])
    high_gray_ = cv2.cvtColor(high_gray_bgr,cv2.COLOR_BGR2YCrCb)

    ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, lower_gray_ ,high_gray_)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    return gray,mask


def getPerspectiveTransformationMatrix():
    corners_pics = np.float32([
            [434,527],[1316,497],
            [-439,899],[2532,818]])

    step=45*10
    corners_real = np.float32( [
            [0,0],[2,0],
            [0,2],[2,2]])*step

    M = cv2.getPerspectiveTransform(corners_pics,corners_real)
    M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
    return M,M_inv,(int(2*step),int(2*step))



def main():
    # inputFolder='/home/nandi/Workspaces/Work/Python/opencvProject/Apps/pics/videos/'
    # inputFolder='C:\\Users\\aki5clj\\Documents\\PythonWorkspace\\Rpi\\Opencv\\LineDetection\\resource\\'
    inputFolder= os.path.realpath('../../resource/videos')
    inputFileName='/move14.h264'
    print(inputFolder+inputFileName)
    frameGenerator=videoRead(inputFolder+inputFileName)
    start=time.time()
    rate=2
    index=0

    M,M_inv,newsize=getPerspectiveTransformationMatrix()
    print(newsize)
    
    lines=[]
    index=0
    nrSlices=15
    for frame,durationTime in frameGenerator:
        frame = cv2.warpPerspective(frame,M,newsize)
        gray,mask=processFrame5(frame)

        if(index==10):
            gray,lines=slidingWindowMethod(gray,mask,nrSlices)
        if(index>=11):
            windowSize=(int(mask.shape[1]*3/nrSlices),int(mask.shape[0]/nrSlices))
            lines=nonslidingWindowMethod(mask,lines,windowSize)
            for line in lines:
                gray=drawLine(gray,line)
                gray=drawWindows(gray,line,windowSize)
            cv2.waitKey()
            
        if(index>100):
            break

        gray=cv2.resize(gray,(int(gray.shape[1]/rate),int(gray.shape[0]/rate)))
        frame = cv2.resize(frame,(int(frame.shape[1]/rate),int(frame.shape[0]/rate)))
        mask = cv2.resize(mask,(int(mask.shape[1]/rate),int(mask.shape[0]/rate)))

        mask = cv2.applyColorMap(mask,cv2.COLORMAP_BONE)
        gray = cv2.applyColorMap(gray,cv2.COLORMAP_BONE)

        vis = np.concatenate((frame,mask,gray), axis=1)

        cv2.imshow('',vis)
        cv2.waitKey(durationTime)

        index+=1
    end=time.time()
    print('sss',(end-start))



if __name__=='__main__':
    main()
