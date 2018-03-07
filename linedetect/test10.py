#!/usr/bin/env python3
import frameProcessor, videoProc, drawFunction,postprocess
import os, cv2 , cv2.plot,time
import numpy as np

from matplotlib import pyplot as plt

import BezierCurve




def getPolynom(line):
    linePoints = BezierCurve.tupleListToComplexList(line)
    minL = np.min(np.imag(linePoints))
    maxL = np.max(np.imag(linePoints))
    X = np.real(linePoints)
    Y = np.imag(linePoints)

    coeff = np.polyfit(Y,X,deg=6) 
    poly = np.poly1d(coeff)
    dpoly = poly.deriv()

    XX = poly(Y)
    lineN = BezierCurve.XYToTuple(XX,Y)

    return lineN,dpoly,[minL,maxL],linePoints

def plot(dPolies,size,limits,lines):



    for dPoly, line in zip(dPolies,lines):
        Y = np.imag(line)
        X = np.real(line)
        nrPoint = len(Y)
        ddX = dPoly(Y)
        dX_deg = np.degrees(np.arctan(ddX))
        
        dY = Y[1:nrPoint] - Y[0:nrPoint-1]
        dX = X[1:nrPoint] - X[0:nrPoint-1]
        dXp = np.degrees(np.arctan(dX/dY))    
        plt.plot(Y,dX_deg)

        plt.plot(Y[0:nrPoint-1],dXp,'--')

    plt.show()


def main():
    print("Test10.py -Main-")
    # source folder
    inputFolder= os.path.realpath('../../resource/videos')
    # source file
    
    # inputFileName='/newRecord/move4.h264'
    inputFileName='/martie2/test3s.h264'

    # inputFileName='/record19Feb/test50L_1.h264'
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
    nrSlices = 15
    windowSize=(int(newSize[1]*2/nrSlices),int(newSize[0]/nrSlices))
    slidingMethod = frameProcessor.SlidingWindowMethod(nrSlice = nrSlices,frameSize=newSize,windowSize = windowSize)
    

    windowSize_nonsliding=(int(newSize[1]*2/nrSlices),int(newSize[0]*2/nrSlices))

    print('Line thinkness is ',2*pxpcm,'[PX]',pxpcm)
    nonslidingMethod = frameProcessor.NonSlidingWindowMethodWithPolynom(windowSize_nonsliding,int(newSize[0]*0.9/nrSlices),2*pxpcm)
    middleGenerator = postprocess.LaneMiddleGenerator(35,pxpcm,newSize,2)
    lineEstimator = postprocess.LineEstimatorBasedPolynom(35,pxpcm,newSize)

    lineVer = postprocess.LaneVerifierBasedDistance(35,pxpcm)
    
    index = 0

    PolynomLines = {}
    middleline = None
    for frame in videoReader.generateFrame():

        t1 = time.time()
        birdview_gray,mask = framelineFilter.apply2(frame)
        mask = drawer.draw(mask)
        
        if index == 0 :
            centerAll,lines = slidingMethod.apply(mask)
            lines = lineVer.checkLane(lines)
            if lines is None:
                continue
            # print(len(lines))
            for index in range(len(lines)):
                line = lines[index]
                newPolyLine = postprocess.PolynomLine(polyDeg)
                newPolyLine.estimatePolynom(line)
                newPolyLine.line = line
                PolynomLines[index]=newPolyLine
                newLine = newPolyLine.generateLinePoint(line)
                if newLine is not None:
                    mask=drawFunction.drawLine(mask,newLine)
            # print (PolynomLines)
            PolynomLines = postprocess.LineOrderCheck(PolynomLines,newSize)
            if len(PolynomLines)<=2:
                nrLine = 3
                nrNewLine = nrLine-len(PolynomLines)
                for key in range(len(PolynomLines)-1,-1,-1):
                    PolynomLines[key+nrNewLine] = PolynomLines[key]
                
                for index in range(0, nrNewLine):
                    PolynomLines[index] = postprocess.PolynomLine(polyDeg)


            
        else:
            # if len(PolynomLines.keys())==2:
            #     PolynomLines[2] =  postprocess.PolynomLine(4)

            lines,PolynomLines = lineEstimator.estimateLine(PolynomLines)
            nonslidingMethod.nonslidingWindowMethod(mask,PolynomLines)
            for key in PolynomLines.keys():
                # print("Key",key,"L",len(PolynomLines[key].line))
                drawFunction.drawWindows(mask,PolynomLines[key].line,windowSize)
            middleline = middleGenerator.generateLine(PolynomLines,middleline)
            
            if middleline is not None:
                mask=drawFunction.drawLine(mask,middleline.line)
            
            # print(PolynomLines)
            for key in PolynomLines.keys():
                if len(PolynomLines[key].line)>3:
                    # drawFunction.drawWindows(mask,PolynomLines[key].line,windowSize)
                    newLine = PolynomLines[key].generateLinePoint(PolynomLines[key].line)
                    if newLine is not None:
                        mask=drawFunction.drawLine(mask,newLine)
            
            
        t2 =time.time()  
        # print('DDD',t2-t1)
        # for line in lines:
        #     # birdview_mask=drawFunction.drawLine(mask,line)
        #     drawFunction.drawWindows(mask,line,windowSize)
        
        # # linesEstimated = lineEstimator.estimateLine(lines)
        # # for line in linesEstimated:
        # #     birdview_mask=drawFunction.drawLine(mask,line)
        # #     drawFunction.drawWindows(mask,line,windowSize)


        # # print(lines)
        # res = drawFunction.drawSubRegion(mask,gray,10,(10,10))
        res = mask
        img_show = res

        cv2.imshow('Frame',img_show)
        index += 1
        
        if cv2.waitKey() & 0xFF == ord('q'):
            break






if __name__ == '__main__':
    main()

