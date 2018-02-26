import numpy as np
import cv2,time,os
import frameProcessor
from matplotlib import pyplot as plt



def generatePicture(thinkness,lineThinkness,angle):
    pics = np.zeros((900,900))

    pics = np.uint8(pics)
    pics = cv2.line(pics,(100,450-thinkness//2),(800,450-thinkness//2),thickness=lineThinkness,color=(255,255,255))
    pics = cv2.line(pics,(100,450+thinkness//2),(800,450+thinkness//2),thickness=lineThinkness,color=(255,255,255))


    M = cv2.getRotationMatrix2D((450,450),angle,1)
    pics = cv2.warpAffine(pics,M,(pics.shape[1],pics.shape[1]))
    
    return pics



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
    
    
    plt.plot(histogram)
    plt.show()        
    return windowscenter


x=np.linspace(0,2,20)
sig=0.4;mu=1
gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
kernel=gauss/np.sum(gauss)


class CannyHistogramProcess:
    def __init__(self,lineThinkness = 13):
        kernel = np.ones((1,9))[0,:]  
        self.smallMedianFilterKernel1 = kernel / np.sum(kernel)
        
        kernel = np.ones((1,21))[0,:]  
        self.bigMedianFilterKernel2 = kernel / np.sum(kernel)
        self.lineThinkness = lineThinkness

    def apply(self,part,yPos):
        # Part size
        part_size = (part.shape[1],part.shape[0]) 
        # Superior limit of white line size
        upperLimitSize = part.shape[0]*8.5
        # Calculating histogram for each coloumn
        histogram = np.sum(part,axis=0)/255 
        # Median 
        Mean = np.mean(histogram)
        # Filtering the histogram with a small median filter
        histogram_f=np.convolve(histogram,self.smallMedianFilterKernel1,'same') 
        # Filtering the histogram with a bigger median filter
        histogram_f1=np.convolve(histogram,self.bigMedianFilterKernel2,'same') 
        histogram_fd=histogram_f1+Mean * 2.0
        # Peak 
        bigger = histogram_f>histogram_fd 

        accumulate=0 
        accumulatePos=0 
        points=[]
    
        biggerI = 0
        nrBigger = 0
        lastSpike = None
        for i in range(part_size[0]): 
        
            if bigger[i] <1 and bigger[i-1] >= 1:
                if nrBigger >=1 :
                    II = int(biggerI/nrBigger)
                else:
                    II = i
                # print("Pike",i,II,accumulate)
                
                if lastSpike is not None:
                    # print ('Spike',i-lastSpike,'Thinkness',thinkness,'accumulate',accumulate,'upperLimitSize',upperLimitSize)
                    if( i -lastSpike > self.lineThinkness - 6 and i - lastSpike < self.lineThinkness + 5) :
                        # print('Line', (i + lastSpike)//2)
                        points.append(((II + lastSpike)//2,yPos))
                        lastSpike = II
                if (accumulate < upperLimitSize):
                    lastSpike = i
                biggerI = 0
                nrBigger = 0
            elif bigger[i]>=1:
                biggerI += i
                nrBigger += 1
            #The non-zero block 
            if histogram_f[i]>1: 
                accumulate+=histogram[i] 
            # The end of a non-zero block 
            elif (histogram_f[i]<=1 and histogram_f[i-1]>1) : 
                accumulate=0 
        return points


class CannyHistogramProcess2:
    def __init__(self,lineThinkness = 13):
        kernel = np.ones((1,9))[0,:]  
        self.smallMedianFilterKernel1 = kernel / np.sum(kernel)
        
        kernel = np.ones((1,21))[0,:]  
        self.bigMedianFilterKernel2 = kernel / np.sum(kernel)
        self.lineThinkness = lineThinkness

        self.std_calc  = StandardDeviationSlidingWindow(40)

    def apply(self,part,yPos):
        # Part size
        part_size = (part.shape[1],part.shape[0]) 
        # Superior limit of white line size
        upperLimitSize = part.shape[0]*8.5
        # Calculating histogram for each coloumn
        histogram = np.sum(part,axis=0)/255 
        # Median 
        # Mean = np.mean(histogram)
        # Filtering the histogram with a small median filter
        histogram_f=np.convolve(histogram,self.smallMedianFilterKernel1,'same') 
        # Filtering the histogram with a bigger median filter
        # histogram_f1=np.convolve(histogram,self.bigMedianFilterKernel2,'same') 
        # histogram_fd=histogram_f1+Mean * 2.0

        str_dev,mean = self.std_calc.apply(histogram_f)
        histogram_fd = str_dev+mean

        # Peak 
        bigger = histogram_f>histogram_fd 

        accumulate=0 
        accumulatePos=0 
        points=[]
    
        biggerI = 0
        nrBigger = 0
        lastSpike = None
        for i in range(part_size[0]): 
        
            if bigger[i] <1 and bigger[i-1] >= 1:
                if nrBigger >=1 :
                    II = int(biggerI/nrBigger)
                else:
                    II = i
                # print("Pike",i,II,accumulate)
                
                if lastSpike is not None:
                    # print ('Spike',i-lastSpike,'Thinkness',thinkness,'accumulate',accumulate,'upperLimitSize',upperLimitSize)
                    if( i -lastSpike > self.lineThinkness - 6 and i - lastSpike < self.lineThinkness + 5) :
                        # print('Line', (i + lastSpike)//2)
                        points.append(((II + lastSpike)//2,yPos))
                        lastSpike = II
                if (accumulate < upperLimitSize):
                    lastSpike = i
                biggerI = 0
                nrBigger = 0
            elif bigger[i]>=1:
                biggerI += i
                nrBigger += 1
            #The non-zero block 
            if histogram_f[i]>1: 
                accumulate+=histogram[i] 
            # The end of a non-zero block 
            elif (histogram_f[i]<=1 and histogram_f[i-1]>1) : 
                accumulate=0 
        return points




    

class StandardDeviationSlidingWindow:
    def __init__(self,window):
        self.kernel =  (np.ones((1,window))/window)[0,:] 
    
    def apply(self,data):
        #  Calculating the mean 
        Mean=np.convolve(data,self.kernel,'same') 
        # Power of the data
        Q = np.power(data,2)
        Q = np.convolve(Q,self.kernel,'same') 
        Mean_power = np.power(Mean,2)
        O = np.sqrt((Q - Mean_power))
        return O,Mean


def chechPoints(mask,points):
    newPoints = []
    for point in points:
        if mask [point[1],point[0]] >0:
            newPoints.append(point)
    return newPoints

def transformPoint(points,MR_inv):
    resPoints = []
    for point in points:
        newpoint = np.int32(np.dot(MR_inv,np.array([[point[0]],[point[1]],[1]])))
        resPoints.append((newpoint[0,0],newpoint[1,0]))
    return resPoints


def getRotationMatrixes(img_size,angles):
    MR = []
    MR_inv = []

    for angle in angles:
        MR_a = cv2.getRotationMatrix2D((img_size[0]/2,img_size[1]/2),angle,1)
        MR_ai = cv2.getRotationMatrix2D((img_size[0]/2,img_size[1]/2),-angle,1)
        MR.append(MR_a)
        MR_inv.append(MR_ai)
    return MR,MR_inv


def main():
    # pics = generatePicture(14,2,90)
    inputFolder= os.path.realpath('../../resource/videos')
    inputFileName='/f_big_50_3.h264'
    # inputFileName='/move2.h264'
    
    cap = cv2.VideoCapture(inputFolder+inputFileName)
    trans = frameProcessor.ImagePersTrans.getPerspectiveTransformation1()

    nrSciles = 30 
    angles = [0,-45,-90]
    img_size= (900,900)

    MR,MR_inv = getRotationMatrixes(img_size,angles)

    hist = CannyHistogramProcess()
    
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = frame
        startTime = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray1 = trans.wrapPerspective(gray)
        edges = cv2.Canny(gray1,threshold1=50, 	threshold2= 150)
        
        # edges1 = cv2.Canny(gray,threshold1=50, 	threshold2= 150)
        # edges1 = trans.wrapPerspective(edges1)

        # cv2.imshow('EE',edges)
        # cv2.imshow('EE1',edges1)
        # cv2.waitKey()
        
        gray_birdView = trans.wrapPerspective(gray)

        pics_size = (edges.shape[1],edges.shape[0])
        
        # mask_gray_birdView = cv2.adaptiveThreshold(gray_birdView,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-13.5)
        # mask_gray_birdView = cv2.dilate(mask_gray_birdView,np.ones((3,3)),iterations=3)

        
        points = [] 
        
        for M_rot, M_rot_inv in zip(MR,MR_inv):
            edges_rot = cv2.warpAffine(edges,M_rot,pics_size)
            for i in range(nrSciles):
                part = edges_rot [ pics_size[1]//nrSciles*i :pics_size[1]//nrSciles*(i+1),: ]
                Tpoints = hist.apply(part,int(pics_size[1]//nrSciles*i+0.5))
                # Tpoints = histogramMethod3(part,int(pics_size[1]//nrSciles*i+0.5),13)
                points += transformPoint(Tpoints,M_rot_inv)

        
        # points = np.int32(points)
        # points = chechPoints(mask_gray_birdView,points)
        
        for point in points:
            # print(point)
            cv2.circle(gray_birdView,point,2,color=(0,255,255))
        endTime = time.time()
        print("Duration",(endTime-startTime))
        cv2.imshow('Lines',gray_birdView)
        if cv2.waitKey() & 0xFF == ord('q'):
            break
    


    



if __name__=="__main__":
    main()