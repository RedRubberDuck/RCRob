import frameProcessor, videoProc, drawFunction,postprocess
import os, cv2 , cv2.plot,time
import numpy as np





def main():
    print("Test10.py -Main-")
    # source folder
    inputFolder= os.path.realpath('../../resource/videos')
    # source file
    
    inputFileName='/move1.h264'
    # inputFileName='/record19Feb2/test50L_1.h264'
    # inputFileName='/f_big_50_2.h264'
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
    lineVer = postprocess.LaneVerifier(29,pxpcm)
    lineEstimator = postprocess.LaneLineEstimator(29,pxpcm)
    # Window size 
    
    index = 0
    for frame in videoReader.generateFrame():

        t1 = time.time()
        gray,birdview_gray,birdview_mask,mask,mask1 = framelineFilter.apply(frame)
        birdview_mask = drawer.draw(mask)
        
        if index == 0 :
            centerAll,lines = slidingMethod.apply(mask)
            # drawFunction.drawWindows(birdview_mask,centerAll,windowSize)
        else:
            lines = nonslidingMethod.nonslidingWindowMethod(mask,lines)
        lines = lineVer.checkLines(lines)
        t2 =time.time()  
        print('DDD',t2-t1)
        for line in lines:
            birdview_mask=drawFunction.drawLine(mask,line)
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

