import cv2
import numpy as np
import PointProcess
import HistogramProcessingFnc
import my as myCpy


class SlicingWindowMethod:
    def __init__(self, nrSlice, frameSize, pxpcm, windowSize):
        self.nrSlice = nrSlice
        self.windowSize = windowSize

        partSize = (frameSize[0], frameSize[1]//nrSlice)
        print('Window size:', windowSize, 'Part size:', partSize)
        # self.histogramProcl1 = HistogramProcessingFnc.HistogramProcessing(0.002777778,0.023570226,lineThinkness =  2*5,xDistanceLimit = windowSize[0]//2,partSize=partSize)
        self.histogramProc = myCpy.HistogramProcessing(
            0.009777778, 0.023570226, partSize[0], partSize[1], 2*pxpcm, windowSize[0]//2)
        # 20,0.002777778,0.023570226,partSize[0],partSize[1],2*5,partSize[1]//2
        self.slicingMethod = myCpy.SlicingMethod(
            nrSlice, 0.002777778, 0.023570226, partSize[0], partSize[1], 2*pxpcm, partSize[1]//2)
        self.liniarityExaminer = PointProcess.LiniarityExaminer(
            inferiorCorrLimit=0.9, lineThinkness=2.2*pxpcm)
        self.pointConnectivity = PointProcess.PointsConnectivity(windowSize)

    def __call__(self, img_bin):
        img_size = (img_bin.shape[1], img_bin.shape[0])

        pointsAll = self.slicingMethod.apply(img_bin)
        pointsAll = np.array(pointsAll)
        # pointsAll = self.pointsLineVerifying(img_bin, pointsAll)
        pointsAll = sorted(pointsAll, key=lambda point: point.imag)

        lines = self.pointConnectivity.connectPoint(pointsAll)
        return pointsAll, lines

    def __call1__(self, img_bin):
        img_size = (img_bin.shape[1], img_bin.shape[0])
        pointsAll = []
        for i in range(self.nrSlice):
            part = img_bin[int(img_size[1]*i/self.nrSlice)                           :int(img_size[1]*(i+1)/self.nrSlice), :]
            yPos = int(img_size[1]*(i+0.5)/self.nrSlice)
            points = self.histogramProc.apply(part, yPos)
            pointsAll += points
        # point_np= np.array(pointsAll,dtype=[('x',np.uint16),('y',np.uint16)])
        pointsAll = self.pointsLineVerifying(img_bin, pointsAll)
        lines = self.pointConnectivity.connectPoint(pointsAll)
        return pointsAll, lines

    def pointsLineVerifying(self, frame, points):
        points_ = []
        frameSize = (frame.shape[1], frame.shape[0])
        for point in points:
            # Window index
            limitsY = np.array([point.imag-self.windowSize[1]//2,
                                point.imag+self.windowSize[1]//2], dtype=np.int32)
            limitsX = np.array([point.real-self.windowSize[0]//2,
                                point.real+self.windowSize[0]//2], dtype=np.int32)
            # Clipping
            limitsX = np.clip(limitsX, 0, frameSize[0])
            limitsY = np.clip(limitsY, 0, frameSize[1])
            #
            part = frame[limitsY[0]:limitsY[1], limitsX[0]:limitsX[1]]

            isline, pointA, nrNonZero = self.liniarityExaminer.examine(
                part, verticalTest=False)

            if isline:
                # print("sdsd")
                points_.append(point)
            # print(points_)
        return points_
