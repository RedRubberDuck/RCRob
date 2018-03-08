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
    inputFileName='/martie8/verylow.h264'

    # inputFileName='/record19Feb2/test50L_1.h264'
    # inputFileName='/record19Feb/test50_8.h264'
    # inputFileName='/f_big_50_4.h264'
    print('Processing:',inputFolder+inputFileName)
    # Video frame reader object
    videoReader = videoProc.VideoReader(inputFolder+inputFileName)
    frameRate = 30.0 
    frameDuration = int(1.0/frameRate*1000)
    

   
    for frame in videoReader.generateFrame():
       
        cv2.imshow('',frame)
        if cv2.waitKey() & 0xFF == ord('q'):
            break

        # index+=1
    end=time.time()

    


if __name__=='__main__':
    main()
