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

    return lineN,dpoly,[minL,maxL]

def plot(dPolies,size,limits):
    # print(limits)
    # lowLimit = np.max(limits[:,0])
    # highLimit = np.min(limits[:,1])
    lowLimit = 0
    highLimit = 400

    print(lowLimit,highLimit)
    Y = np.linspace(lowLimit,highLimit,100)
    Dx_degs = [] 
    for i in range(len(dPolies)):
        Dx = dPolies[i](Y)
        Dx_deg = np.rad2deg(  np.arctan(Dx))
        Dx_degs.append(Dx_deg)
        plt.plot(Y,Dx_deg)
    plt.show()
    
    nrDeg = len(Dx_degs)
    for i in range(nrDeg-1):
        DegI = Dx_degs[i]
        for j in range(i+1,nrDeg):
            DegJ = Dx_degs[j]
            error = 0
            for k in range(len(DegI)):
                error += np.abs(DegI[k] - DegJ[k])
            print('Lines:',i,' ',j,' ',error/len(DegI))


def main():
    print("Test10.py -Main-")
    # source folder
    inputFolder= os.path.realpath('../../resource/videos')
    # source file
    
    inputFileName='/newRecord/move2.h264'
    # inputFileName='/record19Feb2/test50L_5.h264'
    # inputFileName='/f_big_50_.h264'
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
    nrSlices = 15
    windowSize=(int(newSize[1]*2/nrSlices),int(newSize[0]/nrSlices))
    slidingMethod = frameProcessor.SlidingWindowMethod(nrSlice = nrSlices,windowSize = windowSize)
    

    windowSize_nonsliding=(int(newSize[1]*2/nrSlices),int(newSize[0]*2/nrSlices))
    print('Line thinkness is ',2*pxpcm,'[PX]')
    nonslidingMethod = frameProcessor.NonSlidingWindowMethodWithPolynom(windowSize_nonsliding,int(newSize[0]*0.9/nrSlices),2*pxpcm)
    lineVer = postprocess.LaneVerifierBasedDistance(29,pxpcm)
    lineEstimator = postprocess.LaneLineEstimator(29,pxpcm)
    # Window size 
    
    index = 0
    PolynomLines = {}
    for frame in videoReader.generateFrame():

        t1 = time.time()
        gray,birdview_gray,birdview_mask,mask,mask1 = framelineFilter.apply(frame)
        mask = drawer.draw(mask)
        
        if index == 0 :
            centerAll,lines = slidingMethod.apply(mask)
            lines = lineVer.checkLane(lines)
        

            for index in range(len(lines)):
                line = lines[index]
                newPolyLine = postprocess.PolynomLine(2)
                newPolyLine.estimatePolynom(line)
                newPolyLine.line = line
                PolynomLines[index]=newPolyLine
                newLine = newPolyLine.generateLinePoint(line)
                mask=drawFunction.drawLine(mask,newLine)
            PolynomLines = postprocess.LineOrderCheck(PolynomLines,newSize)
            # drawFunction.drawWindows(birdview_mask,centerAll,windowSize)
        else:
            nonslidingMethod.nonslidingWindowMethod(mask,PolynomLines)
            # print(PolynomLines)
            for key in PolynomLines.keys():
                if len(PolynomLines[key].line)>3:
                    newLine = PolynomLines[key].generateLinePoint(PolynomLines[key].line)
                    mask=drawFunction.drawLine(mask,newLine)
            
        
        
        t2 =time.time()  
        print('DDD',t2-t1)
        for line in lines:
            # birdview_mask=drawFunction.drawLine(mask,line)
            drawFunction.drawWindows(mask,line,windowSize)
            

    

        
        # linesEstimated = lineEstimator.estimateLine(lines)
        # for line in linesEstimated:
        #     birdview_mask=drawFunction.drawLine(mask,line)
        #     drawFunction.drawWindows(mask,line,windowSize)


        # print(lines)
        # res = drawFunction.drawSubRegion(mask,gray,10,(10,10))
        res = mask
        img_show = res

        cv2.imshow('Frame',img_show)
        index += 1
        
        if cv2.waitKey() & 0xFF == ord('q'):
            break






if __name__ == '__main__':
    main()

