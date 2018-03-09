#!/usr/bin/env python3
import cv2, time, os
import numpy as np
import time 


import videoProc


def main():
    # source folder
    inputFolder= os.path.realpath('../../resource/videos')
    # source file
    
    # inputFileName='/record20Feb/test6_7.h264'
    inputFileName='/martie8/test1_highResfps10_1.h264'

    # inputFileName='/record19Feb2/test50L_1.h264'
    # inputFileName='/record19Feb/test50_8.h264'
    # inputFileName='/f_big_50_4.h264'
    print('Processing:',inputFolder+inputFileName)
    # Video frame reader object
    videoReader = videoProc.VideoReader(inputFolder+inputFileName)
    frameRate = 30.0 
    frameDuration = int(1.0/frameRate*1000)
    
    newSize = (1640//2,1232//2)    
    videoOutput = cv2.VideoWriter("v1.h264",cv2.VideoWriter_fourcc('H','2','6','4'),10,newSize)


   
    for frame in videoReader.generateFrame():
        resizedFrame = cv2.resize(frame,newSize)
        videoOutput.write(resizedFrame)
        

    videoOutput.release()
    cv2.destroyAllWindows()  

    


if __name__=='__main__':
    main()
