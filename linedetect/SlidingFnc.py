import cv2
import numpy as np
import frameFilter
import PointProcess


class SlidingWindowMethodWithPolynom:
    def __init__(self, windowSize, distanceLimit, lineThinknessPx):
        self.windowSize = windowSize
        self.lineEximiner = PointProcess.LiniarityExaminer(
            inferiorCorrLimit=0.7, lineThinkness=lineThinknessPx)

        self.distanceLimit = distanceLimit
        self.supLimitNrNonZero = np.max(windowSize) * lineThinknessPx * 2.5
        self.infLimitNrNonZero = np.min(windowSize) * lineThinknessPx * 0.3

    def windowCutting(im, pos, windowSize):
        img_size = (im.shape[1], im.shape[0])
        startX = int(pos.real-windowSize[0]/2)
        endX = int(pos.real+windowSize[0]/2)

        startY = int(pos.imag-windowSize[1]/2)
        endY = int(pos.imag+windowSize[1]/2)

        [startX, endX] = np.clip([startX, endX], 0, img_size[0])
        [startY, endY] = np.clip([startY, endY], 0, img_size[1])
        window = im[startY:endY, startX:endX]
        return window, startX, startY

    def simplifyLine(self, line):
        nrPoint = len(line)
        nrRemoved = 0
        for i in range(nrPoint-1):
            pointI = line[i-nrRemoved]
            pointJ = line[i + 1 - nrRemoved]
            # disX = pointJ[0] - pointI[0]
            # disY = pointJ[1] - pointI[1]
            # dis = np.sqrt(disX**2 + disY**2)

            vec = pointJ - pointI
            dis = abs(vec)
            if dis < self.distanceLimit:
                # newPointX = (pointI[0] + pointJ[0])//2
                # newPointY = (pointI[1] + pointJ[1])//2
                # newPoint = (newPointX,newPointY)
                newPoint = (pointI + pointJ)/2
                line[i - nrRemoved] = newPoint
                line.remove(pointJ)
                nrRemoved += 1

    def addFrontPoint(self, mask, imageSize, polynomLine, nrNewPoint=3):
        frontPoint = polynomLine.line[0]
        nrGeneratedPoint = 0
        for i in range(nrNewPoint):
            dV = polynomLine.dPolynom(frontPoint.imag)
            dY = self.distanceLimit*1.0 / np.sqrt(dV**2+1)
            newPointY = int(frontPoint.imag-dY)
            newPointX = int(polynomLine.polynom(newPointY))

            # print(frontPoint)
            # print(newPointX,newPointY)
            # newPoint =

            if (newPointX > 0 and newPointX < imageSize[0]) and (newPointY > 0 and newPointY < imageSize[1]):
                frontPoint = complex(newPointX, newPointY)

                window, startX, startY = SlidingWindowMethodWithPolynom.windowCutting(
                    mask, frontPoint, self.windowSize)
                nrNonZero = cv2.countNonZero(window)
                if nrNonZero > self.infLimitNrNonZero and nrNonZero < self.supLimitNrNonZero:
                    isLine, pointPlus, nrNonZero = self.lineEximiner.examine(
                        window)
                    if isLine:
                        frontPoint = complex(
                            startX+pointPlus[0], startY+pointPlus[1])
                        polynomLine.line.insert(0, frontPoint)
                        nrGeneratedPoint += 1
                    else:
                        # print("Line test ")
                        break
                else:

                    # print("Size test: ",nrNonZero,self.infLimitNrNonZero,self.supLimitNrNonZero)
                    break
            else:
                # print("On the image")
                break

        # if nrGeneratedPoint < nrNewPoint:
        #     print("OOOOO", nrGeneratedPoint)
        return nrGeneratedPoint

    def addBackPoint(self, imageSize, polynomLine, nrNewPoint=3):
        backPoint = polynomLine.line[-1]
        # print(backPoint)
        for i in range(nrNewPoint):
            dV = polynomLine.dPolynom(backPoint.imag)

            dY = self.distanceLimit*1.0 / np.sqrt(dV**2+1)
            # print(dY,dY*dV)
            newPointY = int(backPoint.imag+dY)
            newPointX = int(polynomLine.polynom(newPointY))

            if (newPointX > 0 and newPointX < imageSize[0]) and (newPointY > 0 and newPointY < imageSize[1]):
                backPoint = complex(newPointX, newPointY)
                polynomLine.line.append(backPoint)
                # print(backPoint)
            else:
                break

    def addIntermediatPoint(self, polynomLine):
        line = polynomLine.line
        nrPoint = len(line)
        nrNewPoint = 0
        for i in range(nrPoint-1):
            pointI = line[i+nrNewPoint]
            pointI1 = line[i+1+nrNewPoint]

            # disY = pointI1[1] - pointI[1]
            # disX = pointI1[0] - pointI[0]
            # dist =  np.sqrt(disY**2 + disX**2)
            vec = pointI1 - pointI
            dist = abs(vec)
            nrLineDis = int(dist/self.distanceLimit/1.1)
            # print("NR.Lines:",nrLineDis)
            for j in range(1, nrLineDis):
                newPointY = int(pointI.imag + (vec.imag*j/nrLineDis))
                newPointX = int(polynomLine.polynom(newPointY))
                line.insert(i+j+nrNewPoint, complex(newPointX, newPointY))
            if nrLineDis > 1:
                nrNewPoint += (nrLineDis-1)

    def lineProcess(self, mask, polynomLine):
        line = polynomLine.line
        if len(line) == 0:
            return
        nrFrontGeneratedPoint = 0
        if polynomLine.polynom is not None:
            nrFrontGeneratedPoint = self.addFrontPoint(
                mask, mask.shape, polynomLine)
            self.addBackPoint(mask.shape, polynomLine)
            self.addIntermediatPoint(polynomLine)

        nrPoint = len(line)
        nrRemovedPoint = 0
        # Check all point
        for index in range(nrFrontGeneratedPoint, nrPoint):
            point = line[index - nrRemovedPoint]
            # Copy the surrounding area of the point
            window, startX, startY = SlidingWindowMethodWithPolynom.windowCutting(
                mask, point, self.windowSize)

            nrNonZero = cv2.countNonZero(window)
            # print(nrNonZero
            if nrNonZero > self.infLimitNrNonZero and nrNonZero < self.supLimitNrNonZero:
                isLine, pointPlus, nrNonZero = self.lineEximiner.examine(
                    window)
                if isLine:

                    pointNew = complex(
                        startX+pointPlus[0], startY+pointPlus[1])
                    line[index-nrRemovedPoint] = pointNew
                else:
                    nrRemovedPoint += 1
                    line.remove(point)
            else:
                # print(nrNonZero, self.infLimitNrNonZero,self.supLimitNrNonZero)
                nrRemovedPoint += 1
                line.remove(point)
        # self.addFrontPoint(mask,mask.shape,polynomLine)
        self.simplifyLine(line)
        if len(line) < 3:
            polynomLine.line = []
            return

        polynomLine.estimatePolynom(line)
        # Check the length of the line

    def __call__(self, mask, polynomline_dic):
        img_size = (mask.shape[1], mask.shape[0])
        for polynomline_Key in polynomline_dic:
            self.lineProcess(mask, polynomline_dic[polynomline_Key])
        return
