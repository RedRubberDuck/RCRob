import build.my as my
import numpy as np
import cv2 , cmath
import sys
import cProfile

from matplotlib import pyplot as plt

sys.path.insert(0, "/home/nandi/Workspaces/git/RCRob/linedetect/")
import HistogramProcessingFnc

gray = cv2.imread("img1.jpg")
size = (gray.shape[1],gray.shape[0])
nrSlice = 20
partSize = (size[0],size[1]//nrSlice)
histProcessing = HistogramProcessingFnc.HistogramProcessing(0.002777778,0.023570226,lineThinkness =  2*5,xDistanceLimit = partSize[1]//2,partSize=partSize)

def process(img_bin,nrSlice):
    img_size=(img_bin.shape[1],img_bin.shape[0])
    pointsAll=[]
    for i in range(nrSlice):
        part=img_bin[int(img_size[1]*i/nrSlice):int(img_size[1]*(i+1)/nrSlice),:]
        yPos=int(img_size[1]*(i+0.5)/nrSlice)
        points=histProcessing.apply(part,yPos)
        pointsAll+=points
    return pointsAll


gray = cv2.imread("img1.jpg")
size = (gray.shape[1],gray.shape[0])

cProfile.run('points2 = process(gray[:,:,0],20)')


slicingMethod = my.SlicingMethod(20,0.002777778,0.023570226,partSize[0],partSize[1],2*5,partSize[1]//2)

def process2(gray):
    points = slicingMethod.apply(gray)
    points = sorted(points,key =  lambda point: point.imag)
    return points

cProfile.run('points = process2(gray[:,:,0])')

# print(np.array(points))

plt.figure()
plt.imshow(gray*255)
plt.plot(np.real(points),np.imag(points),'o')
plt.plot(np.real(points2),np.imag(points2),'or')

print(points,'\n\n\n\n\n\n',points2)
plt.show()
