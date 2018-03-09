import cv2
import numpy as np

## Implementing the perspective transformation functionality
class ImagePerspectiveTransformation:
        ## Constructor
        #   @param: M                   The transformation matrix to the new perspective view
        #   @param: M_inv               The matrix of the iverse perspective transformation 
        #   @param: size                The size of the new frame after transformation
        def __init__(self,M,M_inv,size):
                self.M = M
                self.M_inv = M_inv
                self.size = size
        def wrapPerspective(self,frame):
                return cv2.warpPerspective(frame,self.M,self.size)

        # Getting the transformation matrix, the invers transformation matrix, the size of the transformated image for the first camera position
        def getPerspectiveTransformation1():
                corners_pics = np.float32([
                        [434,527],[1316,497],
                        [-439,899],[2532,818]])

                pxpcm = 5
                step=45 * pxpcm
                corners_real = np.float32( [
                        [0,0],[2,0],
                        [0,2],[2,2]])*step

                M = cv2.getPerspectiveTransform(corners_pics,corners_real)
                M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
                return ImagePersTrans(M,M_inv,(int(step*2),int(step*2))),pxpcm
        # Getting the transformation matrix, the invers transformation matrix, the size of the transformated image for the first camera position
        def getPerspectiveTransformation2():
                corners_pics = np.float32([
                        [421,214],[1354,188],
                        [-295,609],[2131,572]])
                # corners_pics /= 2

                pxpcm = 4
                step = 45*pxpcm
                corners_real = np.float32( [
                        [0,0],[2,0],
                        [0,2],[2,2]])*step

                M = cv2.getPerspectiveTransform(corners_pics,corners_real)
                M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
                return ImagePerspectiveTransformation(M,M_inv,(int(step*2),int(step*2))),pxpcm
        def getPerspectiveTransformation3(rate):
                corners_pics = np.float32([
                        [421,214],[1354,188],
                        [-295,609],[2131,572]])//rate
                # corners_pics /= 2

                pxpcm = 2
                step = 45*pxpcm
                corners_real = np.float32( [
                        [0,0],[2,0],
                        [0,2],[2,2]])*step

                M = cv2.getPerspectiveTransform(corners_pics,corners_real)
                M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
                return ImagePerspectiveTransformation(M,M_inv,(int(step*2),int(step*2))),pxpcm