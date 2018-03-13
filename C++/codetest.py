import build.my as my
import numpy as np
import cv2 , cmath
import sys
import cProfile

sys.path.insert(0, "/home/nandi/Workspaces/git/RCRob/linedetect/")


# from matplotlib import pyplot as plt
# import HistogramProcessingFnc

gray = cv2.imread("img1.jpg")
size = (gray.shape[1],gray.shape[0])

part = gray[0:size[0]//20,:]
partSize = (part.shape[1],part.shape[0])

# histProcessing = HistogramProcessingFnc.HistogramProcessing(0.002777778,0.023570226,lineThinkness =  2*5,xDistanceLimit = partSize[1]//2,partSize=partSize)

# cProfile.run('resPy = histProcessing.histogramMethod(part[:,:,0],0)')

# print(resPy)

# histogramProcessing = my.HistogramProcessing(0.002777778,0.023570226,partSize[0],partSize[1],2*5,partSize[1]//2)

# cProfile.run('dst_c = np.array(histogramProcessing.apply(part[:,:,0],0))')
# print(dst_c)


slicingMethod = my.SlicingMethod(100,0.002777778,0.023570226,partSize[0],partSize[1],2*5,partSize[1]//2)
slicingMethod.apply(part[:,:,0])

