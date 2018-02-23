import frameProcessor, videoProc, drawFunction
import os, cv2 , cv2.plot,time
import numpy as np





def main():
    print("Test10.py -Main-")
    # source folder
    inputFolder= os.path.realpath('../../resource/videos')
    # source file
    # inputFileName='/record19Feb2/test50L_5.h264'
    inputFileName='/newRecord/move2.h264'
    # inputFileName = '/f_big_50_1.h264'
    print('Processing:',inputFolder+inputFileName)
    # Video frame reader object
    videoReader = videoProc.VideoReader(inputFolder+inputFileName)
    frameRate = 30.0
    frameDuration = 1.0/frameRate

    # Perspective transformation
    persTransformation = frameProcessor.ImagePersTrans.getPerspectiveTransformation()
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
            frameProcessor.NonSlidingWindowMethod.nonslidingWindowMethod(mask,lines,windowSize)
            
        
        for line in lines:
            birdview_mask=drawFunction.drawLine(mask,line)
            drawFunction.drawWindows(mask,line,windowSize)
        
        t2 =time.time()
        print('DDD',t2-t1)
        
        res = drawFunction.drawSubRegion(mask,gray,20,(10,10))
        img_show = res

        cv2.imshow('Frame',img_show)
        index += 1
        
        if cv2.waitKey() & 0xFF == ord('q'):
            break






if __name__ == '__main__':
    main()

