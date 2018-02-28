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
    
    inputFileName='/move14.h264'
    # inputFileName='/record19Feb2/test50L_5.h264'
    # inputFileName='/f_big_50_1.h264'
    print('Processing:',inputFolder+inputFileName)
    # Video frame reader object
    videoReader = videoProc.VideoReader(inputFolder+inputFileName)
    frameRate = 30.0 
    frameDuration = 1.0/frameRate

    # Perspective transformation
    persTransformation,pxpcm = frameProcessor.ImagePersTrans.getPerspectiveTransformation1()
    # Frame filter to find the line
    framelineFilter = frameProcessor.frameFilter.FrameLineSimpleFilter(persTransformation)
    # Size of the iamge after perspective transformation
    newSize = persTransformation.size
    # Drawer the mask on the corner
    drawer = frameProcessor.frameFilter.TriangleMaksDrawer.cornersMaskPolygons1(newSize)
    # Sliding method 
    nrSlices = 20
    windowSize=(int(newSize[1]*2/nrSlices),int(newSize[0]/nrSlices))
    slidingMethod = frameProcessor.SlidingWindowMethod(nrSlice = nrSlices,windowSize = windowSize)
    

    windowSize_nonsliding=(int(newSize[1]*2/nrSlices),int(newSize[0]*2/nrSlices))
    nonslidingMethod = frameProcessor.NonSlidingWindowMethod(windowSize_nonsliding,int(newSize[0]*0.9/nrSlices))
    lineVer = postprocess.LaneVerifierBasedDistance(29,pxpcm)
    lineEstimator = postprocess.LaneLineEstimator(29,pxpcm)
    # Window size 
    
    index = 0
    polynoms = []
    for frame in videoReader.generateFrame():

        t1 = time.time()
        gray,birdview_gray,birdview_mask,mask,mask1 = framelineFilter.apply(frame)
        birdview_mask = drawer.draw(mask)
        
        if index == 0 :
            centerAll,lines = slidingMethod.apply(mask)
            lines = lineVer.checkLane(lines)
            for line in lines:
                polynom = postprocess.LinePolynom(6)
                polynoms.append(polynom)
        else:
            lines = nonslidingMethod.nonslidingWindowMethod(mask,lines)
        # lines = lineVer.checkLane(lines)
        t2 =time.time()  
        print('DDD',t2-t1)
        for line in lines:
            # birdview_mask=drawFunction.drawLine(mask,line)
            drawFunction.drawWindows(mask,line,windowSize)

        # dPolies = []
        # limits = []
        # LinesComplex = []
        # for i in range(len(lines)):
        #     LinePoly, dPoly, limit,linePoints = getPolynom(lines[i])
        #     LinesComplex . append(linePoints)
        #     limits.append(limit)
        #     dPolies.append(dPoly)
        #     drawFunction.drawLine(mask,LinePoly)
        # limits = np.array(limits)
        # plot(dPolies,newSize,limits,LinesComplex)

        
        # # linesEstimated = lineEstimator.estimateLine(lines)
        # # for line in linesEstimated:
        # #     birdview_mask=drawFunction.drawLine(mask,line)
        # #     drawFunction.drawWindows(mask,line,windowSize)


        # # print(lines)
        # # res = drawFunction.drawSubRegion(mask,gray,10,(10,10))
        res = mask
        img_show = res

        cv2.imshow('Frame',img_show)
        index += 1
        
        if cv2.waitKey() & 0xFF == ord('q'):
            break






if __name__ == '__main__':
    main()

