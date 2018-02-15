import cv2, time, os
from matplotlib import pyplot as plt
import numpy as np

x=np.linspace(0,2,20)
sig=0.4;mu=1
gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
kernel=gauss/np.sum(gauss)

def drawLine(gray,lines):
    nrLines=len(lines)
    for i in range(0,nrLines-1):
        startpoint=lines[i]
        endPoint=lines[i+1]
        cv2.line(gray,startpoint,endPoint,thickness=1,color=(255,255,255))
    return gray

def slidingWindowMethod(gray,mask, nrSlices):
    #filter bog and small objects
    
    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # print(nlabels)
    
    # for labeli in range(1,nlabels)

    img_size=(mask.shape[1],mask.shape[0])
    windowsCenterAll=[]
    windowsCenterSlice=[]
    for i in range(nrSlices):
        part=mask[int(img_size[1]*i/nrSlices):int(img_size[1]*(i+1)/nrSlices),:]
        yPos=int(img_size[1]*(i+0.5)/nrSlices)
        windowsCenter=histogramMethod2(part,yPos)
        windowsCenterAll+=windowsCenter
        windowsCenterSlice.append(windowsCenter)

    centers,lines=postProcessWindowCenter(windowsCenterSlice,windowsCenterAll,img_size[1]/nrSlices)
    for line in lines:
        gray=drawLine(gray,line)
    gray=drawWindows(gray,windowsCenterAll,(int(mask.shape[1]*3/15),int(mask.shape[0]/15)))
    return gray,windowsCenterAll

def postProcessWindowCenter(windowsCenter,windowsCenterAll,yDistance):
    nrCenter=len(windowsCenterAll)
    line=[]
    distanceLimit=100

    center={}
    for i in range(nrCenter):
        centerI=windowsCenterAll[i]
        minDistance=None
        minDistanceJ=0
        for j in range(i,nrCenter):
            centerJ=windowsCenterAll[j]
            if(centerI[1]==centerJ[1]):
                continue
            elif abs(centerI[0]-centerJ[0])<distanceLimit and abs(centerI[1]-centerJ[1])<yDistance*3 and centerI[1]<centerJ[1]:
                distance= np.sqrt((centerI[0]-centerJ[0])**2 +(centerI[1]-centerJ[1])**2)
                if minDistance is None or  minDistance>distance:
                    minDistance=distance
                    minDistanceJ=j
        if minDistance!=None:
            center[centerI]=windowsCenterAll[minDistanceJ]

    centerP=center.copy()
    lines=[]
    currentCenter=list(centerP.keys())[0]
    line=[currentCenter]
    while len(centerP)!=0:
        if currentCenter in centerP:
            dest=centerP[currentCenter]
            line.append(dest)
            del centerP[currentCenter]
            currentCenter=dest
            
        else:
            lines.append(line)
            currentCenter=list(centerP.keys())[0]
            line=[currentCenter]
    lines.append(line)
    return center,lines


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
    #Filter the histogram
    histogram_f=np.convolve(histogram,kernel,'same')

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
                # print(indexP,' ',indexAA)
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

def drawWindows(gray,windowsCenter,windowSize):

    for center in windowsCenter:
        points=np.array([[[center[0]-windowSize[0]/2,center[1]-windowSize[1]/2],
                            [center[0]+windowSize[0]/2,center[1]-windowSize[1]/2],
                            [center[0]+windowSize[0]/2,center[1]+windowSize[1]/2],
                            [center[0]-windowSize[0]/2,center[1]+windowSize[1]/2]]],dtype=np.int32)
        gray=cv2.polylines(gray,points,thickness=1,isClosed=True,color=(255,255,255))
    return gray


def histogramMethod(part):
    part_size=(part.shape[1],part.shape[0])
    histogram=np.sum(part,axis=0)/255
    histogram_f=np.convolve(histogram,kernel,'same')

    maxim=max(histogram_f)
    pp=(histogram_f+maxim*0.1)<histogram

    indexes=[]
    ind=0
    nr=0
    last_index=-1
    for i in range(1,len(pp)):
        if pp[i] == 0 and pp[i-1] == 1:
            p = int(ind/nr)
            if(last_index!=-1 and np.abs(last_index-p)<100):
                p=int((last_index+p)/2)
                indexes[-1]=p
                last_index=p
            else:
                indexes.append(p)
            
            ind=0
            nr=0
            last_index=p
        elif pp[i] == 1:
            ind += i
            nr  += 1
    
    pd=np.zeros((1,len(pp)))
    print(pd.shape)
    for ii in indexes:
        pd[0,ii] = 1
    

    plt.figure()
    plt.subplot(411)
    plt.imshow(part)
    plt.subplot(412)
    plt.plot(gauss)
    plt.subplot(413)
    plt.plot(histogram_f+maxim*0.1)
    plt.plot(histogram_f)
    plt.plot(histogram)
    plt.subplot(414)
    plt.plot(pp)
    # plt.plot(pp_d)
    plt.plot(pd[0,:])
    plt.show()
    



# def histogramMethodOLD(gray,im_bw):
#     img_size=(im_bw.shape[1],im_bw.shape[0])
#     for i in range(nrSlices):
#         part=im_bw[int(img_size[1]*i/nrSlices):int(img_size[1]*(i+1)/nrSlices),:]
#         # cv2.imshow('',part)
#         # cv2.waitKey()    
#         histogramMethod2(part)
# def histogramMethod2(part):
#     histogram=np.sum(part,axis=0)/255
#     l=len(histogram)
    
#     histogram_f=np.convolve(histogram,kernel,'same')
#     maxim=max(histogram_f)
#     pp=(histogram_f+maxim*0.0)<histogram

#     indexes=[]
#     ind=0
#     nr=0
#     last_index=-1
#     accumulate=0
#     for i in range(1,len(pp)):
#         if pp[i] == 0 and pp[i-1] == 1:
            
#             if accumulate < 1000 and accumulate>100:
#                 p = int(ind/nr)   
#                 if(last_index!=-1 and np.abs(last_index-p)<100):
#                     p = int((last_index+p)/2)
#                     indexes[-1] = p
#                     last_index = p
#                 else:
#                     indexes.append(p)
#                 last_index=p
#             accumulate=0
#             ind=0
#             nr=0
#         elif pp[i] == 1:
#             ind += i
#             nr  += 1
#             accumulate += histogram[i]
    
#     # print(indexes)
#     pd=np.zeros((1,len(pp)))
#     # print(pd.shape)
#     for ii in indexes:
#         pd[0,ii] = 10
#         print(' ',ii)
    
#     print(pd)
#     plt.figure()
#     plt.subplot(411)
#     plt.imshow(part)
#     plt.subplot(412)
#     plt.plot(histogram)
#     plt.plot(histogram_f+maxim*0.0)
#     plt.plot(histogram_f)
#     # plt.plot(pp)
#     plt.subplot(413)
#     plt.plot(pp)
#     plt.subplot(414)
#     plt.plot(pd[0,:])

#     plt.show()
    

    

# def histogramMethod(gray,im_bw,nrSlices):
#     img_size=(im_bw.shape[1],im_bw.shape[0])

#     part=im_bw[int(img_size[1]*19/20):img_size[1],:]
    
#     histogram=np.sum(part,axis=0)/255
#     histogram_f=np.convolve(histogram,kernel,'same')
#     maxim=max(histogram_f)
#     pp=(histogram_f+maxim*0.3)<histogram

#     indexes=[]
#     ind=0
#     nr=0
#     last_index=-1
#     for i in range(1,len(pp)):
#         if pp[i] == 0 and pp[i-1] == 1:
#             p = int(ind/nr)
#             if(last_index!=-1 and np.abs(last_index-p)<100):
#                 p=int((last_index+p)/2)
#                 indexes[-1]=p
#                 last_index=p
#             else:
#                 indexes.append(p)
            
#             ind=0
#             nr=0
#             last_index=p
#         elif pp[i] == 1:
#             ind += i
#             nr  += 1

#     pd=np.zeros((1,len(pp)))
#     print(pd.shape)
#     for ii in indexes:
#         pd[0,ii] = 1
#     return indexes



# def slidingWindowMethod(gray,mask,indexes):
#     img_size=(gray.shape[1],gray.shape[0])
#     window_size=(200,25)

#     nrWindows=int(img_size[1]/window_size[1])

#     #draw initial windows
#     windows_centers_out=[]

#     for index in indexes:
#         windows_centers_out.append([(index,int(img_size[1]-window_size[1]/2))])
        
#         points=np.array([[[index-window_size[0]/2,img_size[1]-window_size[1]],
#                             [index+window_size[0]/2,img_size[1]-window_size[1]],
#                             [index+window_size[0]/2,img_size[1]],
#                             [index-window_size[0]/2,img_size[1]]
#                         ]],dtype=np.int32)
#         gray=cv2.polylines(gray,points,thickness=1,isClosed=True,color=(255,255,255))
    
#     ##Searching next windows
#     windows_centerX=indexes
#     nrFalse=np.zeros((1,len(indexes)))
    
#     ##Searching next windows
#     windows_centerX=indexes
#     print(len(indexes))
#     for i in range(1,nrWindows):
#         for win_centerI in range(len(windows_centerX)):
#             win_centerX=windows_centerX[win_centerI]
#             Sum=0
#             Pos=0
#             for j in range(1,50):
#                 colSum1=np.sum(mask[img_size[1]-(i+1)*window_size[1]:img_size[1]-(i)*window_size[1],win_centerX-j])
#                 colSum2=np.sum(mask[img_size[1]-(i+1)*window_size[1]:img_size[1]-(i)*window_size[1],win_centerX+j])
#                 # print(mask[img_size[1]-(i)*window_size[1]:img_size[1]-(i+1)*window_size[1],win_centerX-j])
#                 Pos+=colSum1*(win_centerX-j)+colSum2*(win_centerX+j)
#                 Sum+=colSum1+colSum2
            
#             if(Sum!=0):
#                 windows_centerX[win_centerI]=int(Pos/Sum)
#                 #print!!!!!!!!!!!!!!!!!!!!!!!
#                 points=np.array([[[win_centerX-window_size[0]/2,img_size[1]-(i)*window_size[1]],
#                                 [win_centerX+window_size[0]/2,img_size[1]-(i)*window_size[1]],
#                                 [win_centerX+window_size[0]/2,img_size[1]-(i+1)*window_size[1]],
#                                 [win_centerX-window_size[0]/2,img_size[1]-(i+1)*window_size[1]]
#                                 ]],dtype=np.int32)
#                 gray=cv2.polylines(gray,points,thickness=1,isClosed=True,color=(255,255,255))
#                 winNew=Pos/Sum
#                 points=np.array([[[winNew-window_size[0]/2,img_size[1]-(i)*window_size[1]],
#                                 [winNew+window_size[0]/2,img_size[1]-(i)*window_size[1]],
#                                 [winNew+window_size[0]/2,img_size[1]-(i+1)*window_size[1]],
#                                 [winNew-window_size[0]/2,img_size[1]-(i+1)*window_size[1]]
#                                 ]],dtype=np.int32)
#                 gray=cv2.polylines(gray,points,thickness=1,isClosed=True,color=(255,255,255))

def NonSlidingWindows(gray,mask):

    return gray

#     return gray

def videoRead(fileName):
    cap = cv2.VideoCapture()
    cap.open(fileName)
    # print('OK')
    index=0

    durationTime=int(1.0/30.0*1000.0)

    while (cap.isOpened()):
        ret,frame = cap.read()
        if(ret):
            # cv2.imshow('',frame)
            # cv2.waitKey(durationTime)
            # time.sleep(durationTime/1000.0)
            yield frame,durationTime
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



def processFrame2(frame):
    img_size=(frame.shape[1],frame.shape[0])

    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    hsv[:,:,1] = clahe.apply(hsv[:,:,1])
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])

    lower_white = np.array([0,0,140], dtype=np.uint8)
    upper_white = np.array([255,80,255], dtype=np.uint8)
    # #---------------------------------------------------------------------------

    mask = cv2.inRange(hsv, lower_white, upper_white)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    return gray,mask

def processFrame3(frame):
    img_size=(frame.shape[1],frame.shape[0])

    lower_white_rbg = np.array([[[100,100,100]]], dtype=np.uint8)
    upper_white_rbg = np.array([[[255,255,255]]], dtype=np.uint8)
    mask = cv2.inRange(frame, lower_white_rbg,upper_white_rbg)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    res = gray
    return res,mask


def processFrame4(frame):
    frame=cv2.GaussianBlur(frame,(7,7),0)
    img_size=(frame.shape[1],frame.shape[0])
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(30,30))

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    lower_white_gray = 150
    upper_white_gray = 255
    mask = cv2.inRange(gray, lower_white_gray,upper_white_gray)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    res = gray
    return res,mask

def processFrame5(frame):
    img_size=(frame.shape[1],frame.shape[0])
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(7,7))

    lower_gray_bgr = np.uint8([[[118,118,118]]])
    lower_gray_ = cv2.cvtColor(lower_gray_bgr,cv2.COLOR_BGR2YCrCb)

    high_gray_bgr = np.uint8([[[255,255,255]]])
    high_gray_ = cv2.cvtColor(high_gray_bgr,cv2.COLOR_BGR2YCrCb)

    ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, lower_gray_ ,high_gray_)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    return gray,mask


def getPerspectiveTransformationMatrix():
    corners_pics = np.float32([
            [434,527],[1316,497],
            [-439,899],[2532,818]])

    step=45*10
    corners_real = np.float32( [
            [0,0],[2,0],
            [0,2],[2,2]])*step

    M = cv2.getPerspectiveTransform(corners_pics,corners_real)
    M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
    return M,M_inv,(int(2*step),int(2*step))



def main():
    # inputFolder='/home/nandi/Workspaces/Work/Python/opencvProject/Apps/pics/videos/'
    # inputFolder='C:\\Users\\aki5clj\\Documents\\PythonWorkspace\\Rpi\\Opencv\\LineDetection\\resource\\'
    inputFolder= os.path.realpath('../../resource/videos')
    inputFileName='/f_big_50_4.h264'
    print(inputFolder+inputFileName)
    frameGenerator=videoRead(inputFolder+inputFileName)
    start=time.time()
    rate=2
    index=0

    M,M_inv,newsize=getPerspectiveTransformationMatrix()
    print(newsize)

    index=0
    for frame,durationTime in frameGenerator:
        frame = cv2.warpPerspective(frame,M,newsize)
        gray,mask=processFrame2(frame)

        if(index==10):
            nrSlices=15
            gray,windowsCenter=slidingWindowMethod(gray,mask,nrSlices)
        if(index==11):
             cv2.waitKey()
             break

        gray=cv2.resize(gray,(int(gray.shape[1]/rate),int(gray.shape[0]/rate)))
        frame = cv2.resize(frame,(int(frame.shape[1]/rate),int(frame.shape[0]/rate)))
        mask = cv2.resize(mask,(int(mask.shape[1]/rate),int(mask.shape[0]/rate)))

        mask = cv2.applyColorMap(mask,cv2.COLORMAP_BONE)
        gray = cv2.applyColorMap(gray,cv2.COLORMAP_BONE)

        vis = np.concatenate((frame,mask,gray), axis=1)

        cv2.imshow('',vis)
        cv2.waitKey(durationTime)

        index+=1
    end=time.time()
    print('sss',(end-start))



if __name__=='__main__':
    main()
