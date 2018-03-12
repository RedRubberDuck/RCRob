import cv2
import numpy as np


# Implementing the frame processing method to obtain the white line on the image. It's based on an adaptive threshold method.
# It applies a histogram equalization method, than transform the image to the bird view to have the line a same thinkness. After 
# the transformation it uses a mask to remove the high light reflection on the floor.
class FrameLineSimpleFilter:

    ## Contructor
    #   @param: perspectiveTrans            An object, which has a wrapPerspective function to transfrom the initian image to the another view
    #   @param: ClaheClipLimit              It's a clip limit value, used by the CLACHE method
    #   @param: ClahetileGridSize           It's a grid size, used by the CLACHE method
    #   @param: kernelSize                  It's a kernel size for filtering the image.
    def __init__(self,perspectiveTrans,ClaheClipLimit=3.0,ClahetileGridSize=(46,46),kernelSize=6):
        self.clahe = cv2.createCLAHE(clipLimit=ClaheClipLimit, tileGridSize=ClahetileGridSize)
        self.perspectiveTrans = perspectiveTrans
        self.kernelSize=kernelSize

        self.kernel = cv2.getGaussianKernel(kernelSize,0)
    
    ## Apply function = + conver to Gray + histogramEqualize + perspectiveTransform + filtering +thresholding
    #   @param: frame                       The input frame, which will be processed
    #   @param: useClahe                    It's a flag tobe used the CLACHE or a simple histogram equalisation 
    def apply(self,frame,useClahe=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if (useClahe):
        #     gray = self.clahe.apply(gray)
        # else:
        #      gray = cv2.equalizeHist(gray)
        birdviewGray = self.perspectiveTrans.wrapPerspective(gray)
        # gray = cv2.resize(gray,(birdviewGray.shape[1],birdviewGray.shape[0]))

        imgMasked,Mask1,Mask2=self.filter(birdviewGray)
        return gray,birdviewGray,imgMasked,Mask1,Mask2
    
    ## Filter function = filtering + thresholding
    # It applies a mask to remove the high light reflection
    #   @param: gray                        The gray input frame  
    def filter(self,gray):
        grayf = cv2.GaussianBlur(gray,(self.kernelSize,self.kernelSize),0)

        thres,mask2 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        mask2 = cv2.blur(mask2,(31,31),0)
        thres,mask2 = cv2.threshold(mask2, 140, 255, cv2.THRESH_BINARY)
        mask2 = cv2.dilate(mask2,np.ones((3,3)))
        mask2 = cv2.bitwise_not(mask2)

        mask= cv2.adaptiveThreshold(grayf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,91,-43.5)
        # mask= cv2.adaptiveThreshold(grayf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31*2+1,-13.5*1.5)
        res = cv2.bitwise_and(mask,mask2)
        mask_inv = cv2.bitwise_not(mask)
        mask2_inv = cv2.bitwise_not(mask2)
        res2 = cv2.bitwise_and(mask_inv,mask2_inv)
        res3 = cv2.bitwise_or(res,res2)
        
        return res,mask,mask2

    def grayscale(self,b,g,r):
        print(b,g,r)
        return int(0.299*r+0.587*g+0.114*b)

    def apply2(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        birdviewGray = self.perspectiveTrans.wrapPerspective(gray)
        imgMasked= cv2.adaptiveThreshold(birdviewGray,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,-10.5)
        return birdviewGray,imgMasked


    def apply3(self,frame):
        birdview_color = self.perspectiveTrans.wrapPerspective(frame)
        birdviewGray = cv2.cvtColor(birdview_color, cv2.COLOR_BGR2GRAY)
        mask= cv2.adaptiveThreshold(birdviewGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,-10.5)

        return birdview_color,mask
    

class TriangleMaksDrawer:
    def __init__(self,polygons):
        self.polygons = polygons
    
    def draw(self,frame):
        for polygon in self.polygons:
            cv2.fillPoly(frame, polygon,(0,0,0))
        
        return frame

    def cornersMaskPolygons1(newSize):
        polygons1 = np.array( [[[newSize[1]//1.5,newSize[0]],[newSize[1],newSize[0]//2.4],[newSize[1],newSize[0]]]] ,dtype=np.int32)
        polygons2 = np.array( [[[newSize[1]//6.225,newSize[0]],[0,newSize[0]//1.942],[0,newSize[0]]]] ,dtype=np.int32)
        polygons = [polygons1,polygons2]
        return TriangleMaksDrawer(polygons)






