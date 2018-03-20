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

                pxpcm = 4
                step = 45*pxpcm
                corners_real = np.float32( [
                        [0,0],[2,0],
                        [0,2],[2,2]])*step

                M = cv2.getPerspectiveTransform(corners_pics,corners_real)
                M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
                return ImagePerspectiveTransformation(M,M_inv,(int(step*2),int(step*2))),pxpcm
        def getPerspectiveTransformation4(rate):
                corners_pics = np.float32([
                        [421,214],[1354,188],
                        [-295,609],[2131,572]])//rate
                # corners_pics /= 2

                pxpcm = 4
                step = 45*pxpcm
                offset = 0*pxpcm
                deplase  = -20*pxpcm
                corners_real = np.float32([
                        [0,0],[2,0],
                        [0,2],[2,2]])*step
                corners_real[:,1] = corners_real[:,1] + deplase - offset
                M = cv2.getPerspectiveTransform(corners_pics,corners_real)
                M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)

                # point_blackTriangle = corners_pics
                # # point_blackTriangle[0,0] = 10
                # # point_blackTriangle[3,0] = 10
                # # point_blackTriangle[1,0] = 1648
                # # point_blackTriangle[2,0] = 1648
                # point_blackTriangle = np.transpose(point_blackTriangle)
                # print(point_blackTriangle)
                # point_blackTriangle=np.concatenate((point_blackTriangle,np.ones((1,len(corners_pics)))),axis=0)
                # point_blackTriangle = np.array([point_blackTriangle])
                
                # # point_blackTriangle_newImage=np.dot(point_blackTriangle,M)
                # point_blackTriangle_newImage= np.dot(M_inv,point_blackTriangle)
                # print(point_blackTriangle_newImage)

                # point_blackTriangle[0,:,0] = point_blackTriangle[0,:,0]/point_blackTriangle[0,:,2]
                # point_blackTriangle[0,:,1] = point_blackTriangle[0,:,1]/point_blackTriangle[0,:,2]
                pX1=30
                pY1 = ((pX1 - corners_pics[0,0])/(corners_pics[2,0]-corners_pics[0,0])*(corners_pics[2,1]-corners_pics[0,1]))+corners_pics[0,1]
                
                point1=np.array([pX1,pY1,1])
                point2=np.array([pX1,1232,1])



                pX3=1648-30
                pY3 = ((pX3 - corners_pics[1,0])/(corners_pics[3,0]-corners_pics[1,0])*(corners_pics[3,1]-corners_pics[1,1]))+corners_pics[1,1]
                point3=np.array([pX3,pY3-10,1])
                print(point3)
                point4=np.array([pX3,1232,1])
                
                points = np.array([point1,point3,point4,point2])
                points = np.transpose(points)
                
                
                
                
                res = np.matmul(M,points)
                res = res / res [2,:]
                res = np.transpose(res [:-1])
                print(res)
                
                
                point_blackTriangle_newImage=res
                

                return ImagePerspectiveTransformation(M,M_inv,(int(step*2),int(step*2-offset))),pxpcm,point_blackTriangle_newImage