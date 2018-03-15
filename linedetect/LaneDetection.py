import cv2
import threading

import frameProcessor, videoProc, drawFunction,postprocess, ImageTransformation, SlicingMethod, WindowSlidingFnc

class LaneDetector:

    def __init__(self,rate):
        persTransformation,pxpcm = ImageTransformation.ImagePerspectiveTransformation.getPerspectiveTransformation3(rate)
        self.pxpcm = pxpcm
        self.persTransformation = persTransformation
        self.birdviewImage_size = persTransformation.size
        self.frameFilter = frameProcessor.frameFilter.FrameLineSimpleFilter(persTransformation)
        self.triangleMaskdrawer = frameProcessor.frameFilter.TriangleMaksDrawer.cornersMaskPolygons1(self.birdviewImage_size)

        self.nrSlices = 15
        self.polyDeg = 2
        windowSize=(int(self.birdviewImage_size[1]*2/self.nrSlices),int(self.birdviewImage_size[0]/self.nrSlices))
        self.slicingImageMethod = SlicingMethod.SlicingWindowMethod(nrSlice = self.nrSlices, frameSize = self.birdviewImage_size,pxpcm=self.pxpcm, windowSize = windowSize)
        self.windowSize_sliding = (int(self.birdviewImage_size[1]*2.5/self.nrSlices),int(self.birdviewImage_size[0]*2.5/self.nrSlices))
        self.slidingWindowMethod = WindowSlidingFnc.SlidingWindowMethodWithPolynom(self.windowSize_sliding,int(self.birdviewImage_size[0]*0.9/self.nrSlices),2*pxpcm)
        self.lineVer = postprocess.LaneVerifierBasedDistance(35,pxpcm)
        self.lineEstimator =  postprocess.LineEstimatorBasedPolynom(45,pxpcm,self.birdviewImage_size)
        self.middleGenerator = postprocess.LaneMiddleGenerator(35,pxpcm,self.birdviewImage_size,self.polyDeg)
        
        self.frameNo = -1

        self.PolynomLines = {}
        self.middleline = None
        self.frameProcessMethod = self.slicingMethodFnc
    
    def addLinesToPolinom(self,lines):
        for index in range(len(lines)):
            line = lines[index]
            newPolyLine = postprocess.PolynomLine(self.polyDeg)
            newPolyLine.estimatePolynom(line)
            newPolyLine.line = line
            self.PolynomLines[index]=newPolyLine

    def slicingMethodFnc(self,birdviewGrayFrame):
        centerAll,lines = self.slicingImageMethod(birdviewGrayFrame)
        
        lines = self.lineVer.checkLane(lines)
        self.PolynomLines.clear()
        self.addLinesToPolinom(lines)
        if(len(self.PolynomLines)==0):
            return
        self.PolynomLines = postprocess.LineOrderCheck(self.PolynomLines,self.birdviewImage_size)
        self.CompletePolynom(self.PolynomLines)
        self.middleline = self.middleGenerator.generateLine(self.PolynomLines,self.middleline)
        self.frameProcessMethod = self.slidingMethodFnc


    def slidingMethodFnc(self,birdviewGrayFrame):
        self.PolynomLines = self.lineEstimator.estimateLine(self.PolynomLines)
        self.slidingWindowMethod(birdviewGrayFrame,self.PolynomLines)
        self._checkLines()
        if(len(self.PolynomLines)==0):
            return
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

    def _checkLines(self):
        detectedLines = False
        for key,polyline in self.PolynomLines.items():
            detectedLines = len(polyline.line)!=0 or detectedLines
        if not detectedLines:
            self.PolynomLines.clear()
            self.middleline  = None
            self.frameProcessMethod = self.slicingMethodFnc

    def getMiddleLine(self):
        return self.middleline



class LaneDetectThread(threading.Thread):
    def __init__(self,frameGetterFunc):
        super(LaneDetectThread,self).__init__()
        rate = 2
        self.lanedetect = LaneDetector(rate)
        self.newFrameEvent = threading.Event()
        self.frameGetterFunc = frameGetterFunc
        self.isAlive = False
        self.isActive = False
    
    def start(self):
        self.isAlive = True
        super(LaneDetectThread,self).start()

    def stop(self):
        self.isAlive = False
    
    def activate(self):
        self.isActive = True

    def run(self):
        while(self.isAlive):
            if (self.isActive):
                self.newFrameEvent.wait()
                index,frame =  self.frameGetterFunc()
                self.lanedetect.frameProcess(frame)
                




