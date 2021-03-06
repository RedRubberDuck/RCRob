import cv2
import numpy as np



def drawSubRegion(frame,subframe,rate,pos):
    res_frame = frame.copy()
    frameSize = frame.shape
    if (len(frame.shape)==3 and len(subframe.shape)==3 and frame.shape[2] != subframe.shape[2]) or ( len(frame.shape)!=len(subframe.shape)) :
        print("Number of layer is not equal!")
        return res_frame
        
    subframe = cv2.resize(subframe,(subframe.shape[1]//rate,subframe.shape[0]//rate))
    subframeSize = subframe.shape

    limitX = [pos[0],pos[0]+subframeSize[1]] 
    limitY = [pos[1],pos[1]+subframeSize[0]]

    limitX = np.clip(limitX,0,frameSize[1])
    limitY = np.clip(limitY,0,frameSize[0])
    
    sizeXCopy = limitX[1] - limitX[0]
    sizeYCopy = limitY[1] - limitY[0]

    res_frame[ limitY[0]:limitY[1], limitX[0]:limitX[1] ] = subframe [0:sizeYCopy, 0:sizeXCopy] 
    return res_frame

def drawSubRegionWeigthed(frame,subframe,rate,pos,alpha,beta):
    res_frame = frame.copy()
    frameSize = frame.shape
    if (len(frame.shape)==3 and len(subframe.shape)==3 and frame.shape[2] != subframe.shape[2]) or ( len(frame.shape)!=len(subframe.shape)) :
        print("Number of layer is not equal!")
        return res_frame
        
    subframe = cv2.resize(subframe,(subframe.shape[1]//rate,subframe.shape[0]//rate))
    subframeSize = subframe.shape

    limitX = [pos[0],pos[0]+subframeSize[1]] 
    limitY = [pos[1],pos[1]+subframeSize[0]]

    limitX = np.clip(limitX,0,frameSize[1])
    limitY = np.clip(limitY,0,frameSize[0])
    
    sizeXCopy = limitX[1] - limitX[0]
    sizeYCopy = limitY[1] - limitY[0]

    sumWeight = alpha + beta
    alpha = alpha / sumWeight
    beta = beta / sumWeight

    res_frame[ limitY[0]:limitY[1], limitX[0]:limitX[1] ] = alpha * subframe [0:sizeYCopy, 0:sizeXCopy] + beta * res_frame[ limitY[0]:limitY[1], limitX[0]:limitX[1] ]
    return res_frame


def drawLine(gray,lines):
    nrLines=len(lines)
    for i in range(0,nrLines-1):
        startpoint=lines[i]
        endPoint=lines[i+1]
        
        cv2.line(gray,(startpoint[0],startpoint[1]),(endPoint[0],endPoint[1]),thickness=1,color=(125,255,255))
    return gray

def drawLineComplexNumber(gray,lines):
    nrLines=len(lines)
    for i in range(0,nrLines-1):
        startpoint=lines[i]
        endPoint=lines[i+1]
        startp = (int(startpoint.real),int(startpoint.imag))
        endp = (int(endPoint.real),int(endPoint.imag))
        cv2.line(gray,startp,endp,thickness=1,color=(125,255,255))
    return gray


def drawWindows(gray,windowsCenter,windowSize):
    nrPoint = len(windowsCenter)
    # print(len(windowsCenter))
    for index in range(nrPoint):
        center = windowsCenter[index]
        points=np.array([[[center[0]-windowSize[0]/2,center[1]-windowSize[1]/2],
                            [center[0]+windowSize[0]/2,center[1]-windowSize[1]/2],
                            [center[0]+windowSize[0]/2,center[1]+windowSize[1]/2],
                            [center[0]-windowSize[0]/2,center[1]+windowSize[1]/2]]],dtype=np.int32)
        cv2.polylines(gray,points,thickness=1,isClosed=True,color=(255,255,255))



def drawWindowsComplexNumber(gray,windowsCenter,windowSize):
    nrPoint = len(windowsCenter)
    # print(len(windowsCenter))
    for index in range(nrPoint):
        point = windowsCenter[index]
        points=np.array([[[point.real-windowSize[0]/2,point.imag-windowSize[1]/2],
                            [point.real+windowSize[0]/2,point.imag-windowSize[1]/2],
                            [point.real+windowSize[0]/2,point.imag+windowSize[1]/2],
                            [point.real-windowSize[0]/2,point.imag+windowSize[1]/2]]],dtype=np.int32)
        cv2.polylines(gray,points,thickness=1,isClosed=True,color=(255,255,255))


