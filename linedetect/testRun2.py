#!/usr/bin/env python3

import cv2

import frameProcessor, videoProc, drawFunction,postprocess
import LaneDetection
import os, cv2 , cv2.plot,time
import numpy as np

from matplotlib import pyplot as plt
import BezierCurve

import cProfile

def main():
    
    print("Test10.py -Main-")
    # source folder
    inputFolder= os.path.realpath('../../resource/videos')
    # source file
    
    # inputFileName='/newRecord/move4.h264'
    inputFileName='/martie2/test12.h264'

    print('Processing:',inputFolder+inputFileName)
    # Video frame reader object
    videoReader = videoProc.VideoReader(inputFolder+inputFileName)
    frameRate = 30.0 
    frameDuration = 1.0/frameRate

    # with PyCallGraph(output=graphviz):
    start = time.time()

    laneDetec = LaneDetection.LaneDetector()

    for frame in videoReader.generateFrame():
        laneDetec.frameProcess(frame)
        


    end = time.time()
    print('Runtime:',end-start)




        
if __name__=='__main__':
    cProfile.run('main()')
# 
    # main()