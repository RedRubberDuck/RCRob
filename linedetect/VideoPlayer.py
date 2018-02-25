import numpy as np
import cv2,time,os
import frameProcessor
from matplotlib import pyplot as plt



# def preprocess2(frame,rate):
#     gray = frame

#     kernelsize=31
#     gray = cv2.GaussianBlur(gray,(kernelsize,kernelsize),0)
#     thres,mask2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
#     mask2 = cv2.blur(mask2,(151,151),0)
#     thres,mask2 = cv2.threshold(mask2, 180, 255, cv2.THRESH_BINARY)
#     mask2 = cv2.dilate(mask2,np.ones((50,50)))
#     mask2 = cv2.bitwise_not(mask2)
#     # ADAPTIVE_THRESH_GAUSSIAN_C
#     mask= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,-25)
#     res = cv2.bitwise_and(mask,mask2)
    
#     return res,mask,mask2


# def getPerspectiveTransformationMatrix():
#     corners_pics = np.float32([
#             [434,527],[1316,497],
#             [-439,899],[2532,818]])

#     step=45*10
#     corners_real = np.float32( [
#             [0,0],[2,0],
#             [0,2],[2,2]])*step

#     M = cv2.getPerspectiveTransform(corners_pics,corners_real)
#     M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
#     return M,M_inv,(int(2*step),int(2*step))

def getPerspectiveTransformationMatrix():
    corners_pics = np.float32([
            [434-70,527],[1316+70,497],
            [-439,899],[2532,818]])

    step=45*5
    corners_real = np.float32( [
            [0,0],[2,0],
            [0,2],[2,2]])*step

    M = cv2.getPerspectiveTransform(corners_pics,corners_real)
    M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
    return M,M_inv,(int(2*step),int(2*step))

class ImagePersTrans:
    def __init__(self,M,M_inv,size):
        self.M = M
        self.M_inv = M_inv
        self.size = size
    def wrapPerspective(self,frame):
        return cv2.warpPerspective(frame,self.M,self.size)

class LineFilter:
    def __init__(self,perspectiveTrans,ClaheClipLimit,ClahetileGridSize,kernelSize=21):
        self.clahe = cv2.createCLAHE(clipLimit=ClaheClipLimit, tileGridSize=ClahetileGridSize)
        self.perspectiveTrans = perspectiveTrans
        self.kernelSize=kernelSize

    def apply(self,frame,applyClahe=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (applyClahe):
            gray = self.clahe.apply(gray)
        else:
             gray = cv2.equalizeHist(gray)
        birdviewGray = self.perspectiveTrans.wrapPerspective(gray)
        # gray = cv2.resize(gray,(birdviewGray.shape[1],birdviewGray.shape[0]))

        imgMasked,Mask1,Mask2=self.filter(birdviewGray)
        return birdviewGray,imgMasked,Mask1,Mask2
    
    def filter(self,gray):
        grayf = cv2.GaussianBlur(gray,(self.kernelSize,self.kernelSize),0)

        thres,mask2 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        mask2 = cv2.blur(mask2,(31,31),0)
        thres,mask2 = cv2.threshold(mask2, 180, 255, cv2.THRESH_BINARY)
        mask2 = cv2.dilate(mask2,np.ones((3,3)))
        mask2 = cv2.bitwise_not(mask2)

        mask= cv2.adaptiveThreshold(grayf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-13.5)
        res = cv2.bitwise_and(mask,mask2)
        mask_inv = cv2.bitwise_not(mask)
        mask2_inv = cv2.bitwise_not(mask2)
        res2 = cv2.bitwise_and(mask_inv,mask2_inv)
        res3 = cv2.bitwise_or(res,res2)
        
        return res,mask,mask2

def gkern(l=(5,5), sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ay = np.arange(-l[0] // 2 + 1., l[0] // 2 + 1.)
    ax = np.arange(-l[1] // 2 + 1., l[1] // 2 + 1.)
    yy, xx = np.meshgrid(ay, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel 


def histogramMethod2(part,yPos):
    windowscenter=[]
    part_size=(part.shape[1],part.shape[0])
    #Limit calculation
    slice_size=part_size[1]*part_size[0]
    # upper_limit=0.037037037037037035 
    upper_limit=0.04537037037037035 
    lower_limit=0.009259259259259259

    upperLimitSize = slice_size*upper_limit
    loweLimitSize = slice_size*lower_limit

    #Calculating histogram
    histogram=np.sum(part,axis=0)/255

    Mean=np.sum(histogram)/len(histogram)

    #Filter the histogram
    # kernel =  np.ones((1,41))/41
    kernel =  (np.ones((1,65))/65)[0,:]
    histogram_f=np.convolve(histogram,kernel,'same')
    # histogram_f = histogram

    accumulate=0
    accumulatePos=0
    for i in range(part_size[0]):
        #The non-zero block
        if histogram_f[i]>0:
            accumulate+=histogram_f[i]
            accumulatePos+=histogram_f[i]*i
        # The end of a non-zero block
        elif histogram_f[i]==0 and histogram_f[i-1]>0:
            
            if accumulate<upperLimitSize and accumulate>loweLimitSize:
                #Calculating the middle of the non-zero block
                indexP=int(accumulatePos/accumulate)
                #Verify the distance from the last non-zeros block
                if (len(windowscenter)>0 and abs(windowscenter[-1][0]-indexP)<100):
                    #If the distance is smaller than the threshold, then combines it.
                    indexP=int((windowscenter[-1][0]+indexP)/2)
                    windowscenter[-1]=(indexP,windowscenter[-1][1])
                else:
                    # Add to the list of the windowsCenters
                    windowscenter.append((indexP,yPos))
            accumulate=0
            accumulatePos=0
            
    return windowscenter


def main():
    inputFolder= os.path.realpath('../../resource/videos')
    desFolder = os.path.realpath('../../resource/pic')
    inputFileName='/f_big_50_4.h264'
    # inputFileName='/record19Feb/test50_5.h264'
    inputFileName='/record19Feb2/test50L_3.h264'
    # inputFileName='/move2.h264'
    # inputFileName='/newRecord/move2.h264'
    # inputFileName='/record20Feb/test3_1.h264'
    cap = cv2.VideoCapture(inputFolder+inputFileName)
    M,M_inv,newsize=getPerspectiveTransformationMatrix()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(46,46))
    perspectivTransform = ImagePersTrans(M,M_inv,newsize)
    lineFilter = LineFilter(perspectivTransform,3.0,(46,46))

    trans = frameProcessor.ImagePersTrans.getPerspectiveTransformation1()

    rate=2
    index = 0

    nrSlices = 15

    kernel_gaus = gkern((10,10),sig=1.5)
    kernel_gaus = kernel_gaus/np.sum(kernel_gaus)
    
    kernel = np.zeros((50,50))
    kernel = cv2.circle(kernel,(25,25),12,(255,255,255),thickness=1)
    kernel[25,25] = 255
    kernel = cv2.circle(kernel,(25,25),1,(255,255,255),thickness=2)
    
    # kernel = cv2.filter2D(kernel,-1,kernel_gaus) 

    kernel = kernel / np.sum(kernel)   
    
    plt.imshow(kernel)

    plt.show()

    index += 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray,threshold1=50, 	threshold2= 150)
        # edges = cv2.dilate(edges,np.ones((3,3)),iterations= 1)
        # edges[] = cv2.dilate(edges,np.ones((3,3)),iterations= 1)
        
        gray1 = trans.wrapPerspective(gray)
        # gray1 = gray.copy()
        mask= cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-13.5)
        edges = trans.wrapPerspective(edges)
        
        img_size = (edges.shape[1],edges.shape[0])

        if index>100:
            cv2.imwrite('pp/im'+str(index)+'.jpg',edges)
        index+=1
        print(index)
        
        # for i in range(nrSlices):
        #     part=edges[int(img_size[1]*i/nrSlices):int(img_size[1]*(i+1)/nrSlices),:]
            
        #     hist = np.sum(part,axis=0)
        #     plt.subplot(212)
        #     plt.imshow(part)
        #     plt.subplot(211)
        #     plt.plot(hist)
        #     plt.show()
            
        #     # cv2.imshow('s',part)
        #     # cv2.waitKey()
        # resFinal = np.concatenate((edges,edges_flt,mask_flt,mask3), axis=1)
        # resFinal = cv2.resize(resFinal,(resFinal.shape[1]//rate,resFinal.shape[0]//rate) )


        if(frame is not None):
            cv2.imshow('frame',edges)
        else:
            break
        # # time.sleep(0.01)
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        # if index>:
        #     break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()