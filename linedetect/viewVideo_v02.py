#!/usr/bin/env python3
import cv2, time, os
from matplotlib import pyplot as plt
import numpy as np
import time 


import videoProc, frameProcessor, drawFunction, postprocess, LaneDetection


def main():
    # source folder
    inputFolder= os.path.realpath('../../resource/videos')
    # source file
    
    # inputFileName='/record20Feb/test6_7.h264'
    inputFileName='/martie2/test12.h264'
    # inputFileName='/martie8/test1_highResfps10_1.h264'

    # inputFileName='/record19Feb2/test50L_1.h264'
    # inputFileName='/record19Feb/test50_8.h264'
    # inputFileName='/f_big_50_4.h264'
    print('Processing:',inputFolder+inputFileName)
    # Video frame reader object
    videoReader = videoProc.VideoReaderWithResize(inputFolder+inputFileName,2)
    frameRate = 30.0 
    frameDuration = int(1.0/frameRate*1000)
    

    laneDetec = LaneDetection.LaneDetector()

    for frame in videoReader.readFrame():
        birdview_mask = laneDetec.frameProcess(frame)
        birdview_mask = 255*birdview_mask
        
         
        for key in laneDetec.PolynomLines.keys():
            drawFunction.drawLine(birdview_mask,laneDetec.PolynomLines[key].line)
            drawFunction.drawWindows(birdview_mask,laneDetec.PolynomLines[key].line,laneDetec.windowSize_nonsliding)    
        
        if laneDetec.middleline is not None:
                drawFunction.drawLine(birdview_mask,laneDetec.middleline.line)
    

        birdview_mask=cv2.resize(birdview_mask,(birdview_mask.shape[1]+birdview_mask.shape[1],birdview_mask.shape[0]+birdview_mask.shape[0]))

        cv2.imshow('',birdview_mask)
        if cv2.waitKey() & 0xFF == ord('q'):
            break

        # index+=1
    end=time.time()




if __name__=='__main__':
    main()
