import cv2


import frameProcessor, videoProc, drawFunction,postprocess, ImageTransformation, SlicingMethod

class LaneDetector:

    def __init__(self):
        persTransformation,pxpcm = ImageTransformation.ImagePerspectiveTransformation.getPerspectiveTransformation3()
        self.pxpcm = pxpcm
        self.persTransformation = persTransformation
        self.birdviewImage_size = persTransformation.size
        self.frameFilter = frameProcessor.frameFilter.FrameLineSimpleFilter(persTransformation)
        self.triangleMaskdrawer = frameProcessor.frameFilter.TriangleMaksDrawer.cornersMaskPolygons1(self.birdviewImage_size)

        self.nrSlices = 15
        self.polyDeg = 2
        windowSize=(int(self.birdviewImage_size[1]*2/self.nrSlices),int(self.birdviewImage_size[0]/self.nrSlices))
        self.slidingMethod = SlicingMethod.SlicingWindowMethod(nrSlice = self.nrSlices, frameSize = self.birdviewImage_size, windowSize = windowSize)
        self.windowSize_nonsliding = (int(self.birdviewImage_size[1]*2.5/self.nrSlices),int(self.birdviewImage_size[0]*2.5/self.nrSlices))
        self.nonslidingMethod = frameProcessor.NonSlidingWindowMethodWithPolynom(self.windowSize_nonsliding,int(self.birdviewImage_size[0]*0.9/self.nrSlices),2*pxpcm)
        self.lineVer = postprocess.LaneVerifierBasedDistance(35,pxpcm)
        self.lineEstimator =  postprocess.LineEstimatorBasedPolynom(45,pxpcm,self.birdviewImage_size)
        self.middleGenerator = postprocess.LaneMiddleGenerator(35,pxpcm,self.birdviewImage_size,self.polyDeg)
        self.middleline = None

        self.PolynomLines = {}
        self.middleline = None
        self.frameNo = -1

        
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
        # lines = self.lineVer.checkLane(lines)
        self.PolynomLines.clear()
        self.addLinesToPolinom(lines)
        self.PolynomLines = postprocess.LineOrderCheck(self.PolynomLines,self.birdviewImage_size)
        self.CompletePolynom(self.PolynomLines)
        self.frameProcessMethod = self.nonslidingMethodFnc


    def nonslidingMethodFnc(self,birdviewGrayFrame):
        self.PolynomLines = self.lineEstimator.estimateLine(self.PolynomLines)
        self.nonslidingMethod.nonslidingWindowMethod(birdviewGrayFrame,self.PolynomLines)
        self.middleline = self.middleGenerator.generateLine(self.PolynomLines,self.middleline)


    def frameProcess(self,frame):
        birdview_gray,birdview_mask = self.frameFilter.apply2(frame)
        birdview_mask = self.triangleMaskdrawer.draw(birdview_mask)
        self.frameProcessMethod(birdview_mask)
        return birdview_mask

    
    def CompletePolynom(self,PolynomLines):
        if len(PolynomLines)<=2:
            nrLine = 3
            nrNewLine = nrLine-len(PolynomLines)
            for key in range(len(PolynomLines)-1,-1,-1):
                PolynomLines[key+nrNewLine] = PolynomLines[key]
            
            for index in range(0, nrNewLine):
                PolynomLines[index] = postprocess.PolynomLine(self.polyDeg)


