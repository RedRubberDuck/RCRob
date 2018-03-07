import cv2


import frameProcessor, videoProc, drawFunction,postprocess

class LaneDetector:

    def __init__(self):
        persTransformation,pxpcm = frameProcessor.ImagePersTrans.getPerspectiveTransformation2()
        self.pxpcm = pxpcm
        self.persTransformation = persTransformation
        self.birdviewImage_size = persTransformation.size
        self.frameFilter = frameProcessor.frameFilter.FrameLineSimpleFilter(persTransformation)
        self.triangleMaskdrawer = frameProcessor.frameFilter.TriangleMaksDrawer.cornersMaskPolygons1(self.birdviewImage_size)

        self.nrSlices = 20
        windowSize=(int(self.birdviewImage_size[1]*2/self.nrSlices),int(self.birdviewImage_size[0]/self.nrSlices))
        self.slidingMethod = frameProcessor.SlidingWindowMethod(nrSlice = self.nrSlices, frameSize = self.birdviewImage_size, windowSize = windowSize)
        self.windowSize_nonsliding = (int(self.birdviewImage_size[1]*2/self.nrSlices),int(self.birdviewImage_size[0]*2/self.nrSlices))
        self.nonslidingMethod = frameProcessor.NonSlidingWindowMethodWithPolynom(self.windowSize_nonsliding,int(self.birdviewImage_size[0]*0.9/self.nrSlices),2*pxpcm)
        self.PolynomLines = {}
        self.middleline = None
        self.frameNo = -1

        self.polyDeg = 2
        self.frameProcessMethod = self.slidingMethodFnc

        
    
    def addLinesToPolinom(self,lines):
        for index in range(len(lines)):
            line = lines[index]
            newPolyLine = postprocess.PolynomLine(self.polyDeg)
            newPolyLine.estimatePolynom(line)
            newPolyLine.line = line
            self.PolynomLines[index]=newPolyLine

    def slidingMethodFnc(self,birdviewGrayFrame):
        centerAll,lines = self.slidingMethod.apply(birdviewGrayFrame)
        self.PolynomLines.clear()
        self.addLinesToPolinom(lines)
        self.frameProcessMethod = self.nonslidingMethodFnc

    def nonslidingMethodFnc(self,birdviewGrayFrame):
        self.nonslidingMethod.nonslidingWindowMethod(birdviewGrayFrame,self.PolynomLines)



    def frameProcess(self,frame):
        birdview_gray,birdview_mask = self.frameFilter.apply2(frame)
        birdview_mask = self.triangleMaskdrawer.draw(birdview_mask)
        self.frameProcessMethod(birdview_mask)
        return birdview_mask


