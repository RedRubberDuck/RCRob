#!/usr/bin/env python3

import cv2

import frameProcessor, videoProc, drawFunction,postprocess
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

    # inputFileName='/record19Feb2/test50L_1.h264'
    # inputFileName='/record19Feb/test50_8.h264'
    # inputFileName='/f_big_50_4.h264'
    print('Processing:',inputFolder+inputFileName)
    # Video frame reader object
    videoReader = videoProc.VideoReader(inputFolder+inputFileName)
    frameRate = 30.0 
    frameDuration = 1.0/frameRate
    polyDeg = 2

    # Perspective transformation
    persTransformation,pxpcm = frameProcessor.ImagePersTrans.getPerspectiveTransformation2()
    # Frame filter to find the line
    framelineFilter = frameProcessor.frameFilter.FrameLineSimpleFilter(persTransformation)
    # Size of the iamge after perspective transformation
    newSize = persTransformation.size
    print(newSize)
    # Drawer the mask on the corner
    drawer = frameProcessor.frameFilter.TriangleMaksDrawer.cornersMaskPolygons1(newSize)
    # Sliding method 
    nrSlices = 20
    windowSize=(int(newSize[1]*2/nrSlices),int(newSize[0]/nrSlices))
    slidingMethod = frameProcessor.SlidingWindowMethod(nrSlice = nrSlices, frameSize = newSize, windowSize = windowSize)
    

    windowSize_nonsliding=(int(newSize[1]*2/nrSlices),int(newSize[0]*2/nrSlices))

    print('Line thinkness is ',2*pxpcm,'[PX]',pxpcm)
    nonslidingMethod = frameProcessor.NonSlidingWindowMethodWithPolynom(windowSize_nonsliding,int(newSize[0]*0.9/nrSlices),2*pxpcm)
    middleGenerator = postprocess.LaneMiddleGenerator(35,pxpcm,newSize,2)
    
    lineEstimator = postprocess.LineEstimatorBasedPolynom(35,pxpcm,newSize)
    lineVer = postprocess.LaneVerifierBasedDistance(35,pxpcm)
    
    index = 0

    PolynomLines = {}
    middleline = None


    # with PyCallGraph(output=graphviz):
    start = time.time()

    for frame in videoReader.generateFrame():
        birdview_gray,birdview_mask = framelineFilter.apply2(frame)
        
        # It can be commanted some case. 
        birdview_mask = drawer.draw(birdview_mask)

        if index == 0 :
            centerAll,lines = slidingMethod.apply(birdview_mask)
            for index in range(len(lines)):
                line = lines[index]
                newPolyLine = postprocess.PolynomLine(polyDeg)
                newPolyLine.estimatePolynom(line)
                newPolyLine.line = line
                PolynomLines[index]=newPolyLine
        else:
            nonslidingMethod.nonslidingWindowMethod(birdview_mask,PolynomLines)

        
        index += 1 
        # endttt = time.time()
        # print("s",endttt-starttt)



    end = time.time()
    print('Runtime:',end-start)




        
if __name__=='__main__':
    cProfile.run('main()')
# 
    # main()