import cv2, time, os
from matplotlib import pyplot as plt
import numpy as np
import time 

import videoProc, frameProcessor


def main():
    # inputFolder='/home/nandi/Workspaces/Work/Python/opencvProject/Apps/pics/videos/'
    # inputFolder='C:\\Users\\aki5clj\\Documents\\PythonWorkspace\\Rpi\\Opencv\\LineDetection\\resource\\'
    inputFolder= os.path.realpath('../../resource/videos')
    # inputFileName='/record19Feb2/test50L_5.h264'
    inputFileName='/martie2/test1.h264'
    # inputFileName='/newRecord/move14.h264'
    # inputFileName = '/f_big_50_2.h264'
    print(inputFolder+inputFileName)
    
    videoReader = videoProc.VideoReader(inputFolder+inputFileName)
    frameRate = 30
    frameDuration = int(1/frameRate*1000)
    

    start=time.time()
    rate=2
    index=0

    persTransformation,pxpcm=frameProcessor.ImagePersTrans.getPerspectiveTransformation2()
    framelineFilter = frameProcessor.frameFilter.FrameLineSimpleFilter(persTransformation)
    # print(newsize)
    
    lines=[]
    index=0
    nrSlices=15
    startIndex=10
    for frame in videoReader.generateFrame():

        gray,birdview_gray,birdview_mask,mask,mask1 = framelineFilter.apply(frame)
        # frame = persTransformation.wrapPerspective(frame)

        # # frame = cv2.GaussianBlur(frame
        # # ,(3,3),10)
        # gray,mask=processFrame2(frame)
        # # gray,mask=processFrame3(frame)
        # windowSize=(int(mask.shape[1]*3/nrSlices),int(mask.shape[0]/nrSlices))
        

        # if(index>startIndex):
        #     lines=nonslidingWindowMethod(mask,lines,windowSize)
        #     for line in lines:
        #         gray=drawLine(gray,line)
        #         gray=drawWindows(gray,line,windowSize)

        
        # if(index==startIndex):
        #     gray,lines=slidingWindowMethod(gray,mask,windowSize,nrSlices)
        #     for line in lines:
        #         gray=drawLine(gray,line)
        #         gray=drawWindows(gray,line,windowSize)    
        
        # gray=cv2.resize(gray,(int(gray.shape[1]/rate),int(gray.shape[0]/rate)))
        # birdview_mask = cv2.resize(frame,(int(frame.shape[1]/rate),int(frame.shape[0]/rate)))
        # mask = cv2.resize(mask,(int(mask.shape[1]/rate),int(mask.shape[0]/rate)))

        # mask = cv2.applyColorMap(mask,cv2.COLORMAP_BONE)
        # gray = cv2.applyColorMap(gray,cv2.COLORMAP_BONE)

        # vis = np.concatenate((frame,mask,gray), axis=1)

        cv2.imshow('',birdview_mask)
        if cv2.waitKey(frameDuration) & 0xFF == ord('q'):
            break

        index+=1
    end=time.time()
    print('sss',(end-start))



if __name__=='__main__':
    main()
