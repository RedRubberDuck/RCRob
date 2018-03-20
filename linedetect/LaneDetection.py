import cv2
import threading

import videoProc
import drawFunction
import ImageTransformation
import SlicingMethod
from  SlidingFnc import *
from frameFilter import *
from LaneVerifierBasedDistance import *
from LineEstimatorBasedPolynom import *
from LaneMiddleGenerator import *
from PolynomLine import *
from MyUtility import LineOrderCheck
import cProfile
import drawFunction


class LaneDetector:

    def __init__(self, rate):
        persTransformation, pxpcm,point_backTriangle = ImageTransformation.ImagePerspectiveTransformation.getPerspectiveTransformation4(
            rate)
        self.pxpcm = pxpcm
        self.persTransformation = persTransformation
        self.birdviewImage_size = persTransformation.size
        self.frameFilter = FrameLineSimpleFilter(
            persTransformation)
        self.triangleMaskdrawer = TriangleMaksDrawer.cornersMaskPolygons2(
            self.birdviewImage_size,point_backTriangle)

        self.nrSlices = 15
        self.polyDeg = 2
        windowSize = (int(self.birdviewImage_size[1]*2/self.nrSlices), int(
            self.birdviewImage_size[0]/self.nrSlices))
        self.slicingImageMethod = SlicingMethod.SlicingWindowMethod(
            nrSlice=self.nrSlices, frameSize=self.birdviewImage_size, pxpcm=self.pxpcm, windowSize=windowSize)
        self.windowSize_sliding = (int(
            self.birdviewImage_size[1]*2.5/self.nrSlices), int(self.birdviewImage_size[0]*2.5/self.nrSlices))
        self.slidingWindowMethod = SlidingWindowMethodWithPolynom(
            self.windowSize_sliding, int(self.birdviewImage_size[0]*0.5/self.nrSlices), 2*pxpcm)
        self.lineVer = LaneVerifierBasedDistance(35, pxpcm)
        self.lineEstimator = LineEstimatorBasedPolynom(
            45, pxpcm, self.birdviewImage_size)
        self.middleGenerator = LaneMiddleGenerator(
            35, pxpcm, self.birdviewImage_size, self.polyDeg)

        self.frameNo = -1

        self.PolynomLines = {}
        self.middleline = None
        self.frameProcessMethod = self.slicingMethodFnc

        # Temporary variable for debug
        self.birdviewMask = None

    def addLinesToPolinom(self, lines):
        for index in range(len(lines)):
            line = lines[index]
            newPolyLine = PolynomLine(self.polyDeg)
            newPolyLine.estimatePolynom(line)
            newPolyLine.line = line
            self.PolynomLines[index] = newPolyLine

    def slicingMethodFnc(self, birdviewGrayFrame):
        centerAll, lines = self.slicingImageMethod(birdviewGrayFrame)

        lines = self.lineVer.checkLane(lines)
        self.PolynomLines.clear()
        self.addLinesToPolinom(lines)
        if(len(self.PolynomLines) == 0):
            return
        self.PolynomLines = LineOrderCheck(
            self.PolynomLines, self.birdviewImage_size)
        self.CompletePolynom(self.PolynomLines)
        self.middleline = self.middleGenerator.generateLine(
            self.PolynomLines, self.middleline)
        self.frameProcessMethod = self.slidingMethodFnc

    def slidingMethodFnc(self, birdviewGrayFrame):
        self.PolynomLines = self.lineEstimator.estimateLine(self.PolynomLines)
        self.slidingWindowMethod(birdviewGrayFrame, self.PolynomLines)
        self._checkLines()
        if(len(self.PolynomLines) == 0):
            return
        self.middleline = self.middleGenerator.generateLine(
            self.PolynomLines, self.middleline)
        self.linesOriantation()

    def linesOriantation(self):
        der = 0
        nrSection = 0 
        for key in self.PolynomLines:
            polyLine = self.PolynomLines[key]
            if polyLine is not None and polyLine.line is not None:
                for i in range(1,len(polyLine.line)):
                    der += polyLine.line[i]-polyLine.line[i-1]
                    nrSection += 1
        derMedie = der/nrSection
        print('frame orinatation',derMedie,np.degrees(np.arctan( derMedie.real/derMedie.imag)))



    def frameProcess(self, frame):
        birdview_gray, birdview_mask = self.frameFilter.apply2(frame)
        self.birdviewMask = birdview_mask
        birdview_mask = self.triangleMaskdrawer.draw(birdview_mask)
        birdview_gray = self.triangleMaskdrawer.draw(birdview_gray)
        self.frameProcessMethod(birdview_mask)
        return birdview_mask,birdview_gray

    def CompletePolynom(self, PolynomLines):
        if len(PolynomLines) <= 2:
            nrLine = 3
            nrNewLine = nrLine-len(PolynomLines)
            for key in range(len(PolynomLines)-1, -1, -1):
                PolynomLines[key+nrNewLine] = PolynomLines[key]

            for index in range(0, nrNewLine):
                PolynomLines[index] = PolynomLine(self.polyDeg)

    def _checkLines(self):
        detectedLines = False
        for key, polyline in self.PolynomLines.items():
            detectedLines = len(polyline.line) != 0 or detectedLines
        if not detectedLines:
            self.PolynomLines.clear()
            self.middleline = None
            self.frameProcessMethod = self.slicingMethodFnc

    def getMiddleLine(self):
        return self.middleline

    def getDistanceFromMiddleLine(self):
        distance = 45 * self.pxpcm
        if (self.middleline is not None and self.middleline.polynom is not None):
            Xpx = self.middleline.polynom(self.birdviewImage_size[1]-distance)
            distanceXpx = Xpx-self.birdviewImage_size[0]/2
            return distanceXpx/self.pxpcm
        return None


class LaneDetectThread(threading.Thread):
    def __init__(self, frameGetterFunc, rate):
        super(LaneDetectThread, self).__init__()
        # rate = 2
        self.lanedetect = LaneDetector(rate)
        self.newFrameEvent = threading.Event()
        self.frameGetterFunc = frameGetterFunc
        self.isAlive = False
        self.isActive = False
        self.lastInamgeNo = -1

    def start(self):
        self.isAlive = True
        super(LaneDetectThread, self).start()

    def stop(self):
        self.isAlive = False

    def activate(self):
        self.isActive = True
        self.newFrameEvent.clear()

    def run(self):
        cProfile.runctx('self._run()', globals(), locals())

    def draw(self, birdview_mask):
        for key in self.lanedetect.PolynomLines.keys():
            drawFunction.drawLineComplexNumber(
                birdview_mask, self.lanedetect.PolynomLines[key].line)
            drawFunction.drawWindowsComplexNumber(
                birdview_mask, self.lanedetect.PolynomLines[key].line, self.lanedetect.windowSize_sliding)

    def _run(self):
        while(self.isAlive):
            if (self.isActive and self.newFrameEvent.wait(0.01)):
                index, frame = self.frameGetterFunc()
                print("Image no.:", index)

                self.newFrameEvent.clear()
                self.lanedetect.frameProcess(frame)
                if self.lanedetect.getDistanceFromMiddleLine() is not None:
                    self.lastInamgeNo = index
                else:
                    birdviewMask = self.lanedetect.birdviewMask*255
                    cv2.imwrite("im"+str(index)+".jpg",
                                birdviewMask)

    def getDistanceFromOptimelLine(self):
        distanceX = self.lanedetect.getDistanceFromMiddleLine()
        return distanceX

    def getFrameId(self):
        return self.lastInamgeNo
