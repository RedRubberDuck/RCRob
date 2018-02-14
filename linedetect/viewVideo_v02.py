import cv2,numpy,time, os
from matplotlib import pyplot as plt

x=numpy.linspace(0,2,100)
sig=0.3;mu=1
gauss = numpy.exp(-numpy.power(x - mu, 2.) / (2 * numpy.power(sig, 2.)))
kernel=gauss/numpy.sum(gauss)

def slidingWindowMethod(im_bw,nrSlices):
    img_size=(im_bw.shape[1],im_bw.shape[0])
    for i in range(nrSlices):
        part=im_bw[int(img_size[1]*i/nrSlices):int(img_size[1]*(i+1)/nrSlices),:]
        # cv2.imshow('',part)
        # cv2.waitKey()    
        histogramMethod2(part)
def histogramMethod2(part):
    histogram=numpy.sum(part,axis=0)/255
    l=len(histogram)
    
    histogram_f=numpy.convolve(histogram,kernel,'same')
    maxim=max(histogram_f)
    pp=(histogram_f+maxim*0.0)<histogram

    indexes=[]
    ind=0
    nr=0
    last_index=-1
    accumulate=0
    for i in range(1,len(pp)):
        if pp[i] == 0 and pp[i-1] == 1:
            
            if accumulate < 1000 and accumulate>100:
                p = int(ind/nr)   
                if(last_index!=-1 and numpy.abs(last_index-p)<100):
                    p = int((last_index+p)/2)
                    indexes[-1] = p
                    last_index = p
                else:
                    indexes.append(p)
                last_index=p
            accumulate=0
            ind=0
            nr=0
        elif pp[i] == 1:
            ind += i
            nr  += 1
            accumulate += histogram[i]
    
    # print(indexes)
    pd=numpy.zeros((1,len(pp)))
    # print(pd.shape)
    for ii in indexes:
        pd[0,ii] = 10
        print(' ',ii)
    
    print(pd)
    plt.figure()
    plt.subplot(411)
    plt.imshow(part)
    plt.subplot(412)
    plt.plot(histogram)
    plt.plot(histogram_f+maxim*0.0)
    plt.plot(histogram_f)
    # plt.plot(pp)
    plt.subplot(413)
    plt.plot(pp)
    plt.subplot(414)
    plt.plot(pd[0,:])

    plt.show()
    

    

def histogramMethod(gray,im_bw,nrSlices):
    img_size=(im_bw.shape[1],im_bw.shape[0])

    part=im_bw[int(img_size[1]*19/20):img_size[1],:]
    
    histogram=numpy.sum(part,axis=0)/255
    histogram_f=numpy.convolve(histogram,kernel,'same')
    maxim=max(histogram_f)
    pp=(histogram_f+maxim*0.3)<histogram

    indexes=[]
    ind=0
    nr=0
    last_index=-1
    for i in range(1,len(pp)):
        if pp[i] == 0 and pp[i-1] == 1:
            p = int(ind/nr)
            if(last_index!=-1 and numpy.abs(last_index-p)<100):
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

    pd=numpy.zeros((1,len(pp)))
    print(pd.shape)
    for ii in indexes:
        pd[0,ii] = 1
    return indexes



# def slidingWindowMethod(gray,mask,indexes):
#     img_size=(gray.shape[1],gray.shape[0])
#     window_size=(200,25)
#     nrWindows=int(img_size[1]/window_size[1])

#     #draw initial windows
#     windows_centers_out=[]
#     for index in indexes:
#         windows_centers_out.append([(index,int(img_size[1]-window_size[1]/2))]) 
    
#     ##Searching next windows
#     windows_centerX=indexes
#     nrFalse=numpy.zeros((1,len(indexes)))
    
#     for i in range(1,nrWindows):
#         for win_centerI in range(len(windows_centerX)):
#             if(nrFalse[0,win_centerI]>5):
#                 continue

#             win_centerX=windows_centerX[win_centerI]
#             Sum=0
#             PosAcumulate=0
#             for j in range(1,50):
#                 if(win_centerX-j>=0):
#                     colSum1=numpy.sum(mask[img_size[1]-(i+1)*window_size[1]:img_size[1]-(i)*window_size[1],win_centerX-j])
#                 else:
#                     colSum1=0
#                 if(win_centerX+j<=img_size[0]):
#                     colSum2=numpy.sum(mask[img_size[1]-(i+1)*window_size[1]:img_size[1]-(i)*window_size[1],win_centerX+j])
#                 else:
#                     colSum2=0
#                 PosAcumulate+=colSum1*(win_centerX-j)+colSum2*(win_centerX+j)
#                 Sum+=colSum1+colSum2
            
#             if(Sum!=0):
#                 windows_centerX[win_centerI]=int(PosAcumulate/Sum)
#                 pos=(int(PosAcumulate/Sum),int(img_size[1]-(i+0.5)*window_size[1]))
#                 windows_centers_out[win_centerI].append(pos)
#             else:
#                 nrFalse[0,win_centerI]+=1
#     return windows_centers_out

def NonSlidingWindows(gray,mask):

    return gray

def drawWindows(winCenters,gray):
    window_size=(200,25)
    for winCenterLine in winCenters:
        for winCenter in  winCenterLine:
            points=numpy.array([[[winCenter[0]-window_size[0]/2,winCenter[1]-window_size[1]/2],
                                [winCenter[0]+window_size[0]/2,winCenter[1]-window_size[1]/2],
                                [winCenter[0]+window_size[0]/2,winCenter[1]+window_size[1]/2],
                                [winCenter[0]-window_size[0]/2,winCenter[1]+window_size[1]/2]
                                ]],dtype=numpy.int32)
            gray=cv2.polylines(gray,points,thickness=1,isClosed=True,color=(255,255,255))
    
    return gray

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

    lower_white = numpy.array([0,0,140], dtype=numpy.uint8)
    upper_white = numpy.array([255,80,255], dtype=numpy.uint8)
    # #---------------------------------------------------------------------------

    mask = cv2.inRange(hsv, lower_white, upper_white)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    return gray,mask

def processFrame3(frame):
    img_size=(frame.shape[1],frame.shape[0])

    lower_white_rbg = numpy.array([[[100,100,100]]], dtype=numpy.uint8)
    upper_white_rbg = numpy.array([[[255,255,255]]], dtype=numpy.uint8)
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

    lower_gray_bgr = numpy.uint8([[[118,118,118]]])
    lower_gray_ = cv2.cvtColor(lower_gray_bgr,cv2.COLOR_BGR2YCrCb)

    high_gray_bgr = numpy.uint8([[[255,255,255]]])
    high_gray_ = cv2.cvtColor(high_gray_bgr,cv2.COLOR_BGR2YCrCb)

    ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, lower_gray_ ,high_gray_)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    return gray,mask


def getPerspectiveTransformationMatrix():
    corners_pics = numpy.float32([
            [434,527],[1316,497],
            [-439,899],[2532,818]])

    step=45*10
    corners_real = numpy.float32( [
            [0,0],[2,0],
            [0,2],[2,2]])*step

    M = cv2.getPerspectiveTransform(corners_pics,corners_real)
    M_inv = cv2.getPerspectiveTransform(corners_real,corners_pics)
    return M,M_inv,(int(2*step),int(2*step))



def main():
    # inumpyutFolder='/home/nandi/Workspaces/Work/Python/opencvProject/Apps/pics/videos/'
    inumpyutFolder=os.path.realpath('../../resource/videos')
    inumpyutFileName='/f_big_50_1.h264'
    frameGenerator=videoRead(inumpyutFolder+inumpyutFileName)
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
            slidingWindowMethod(mask,20)
            indexes=histogramMethod(gray,mask,9)
            # winCenters=slidingWindowMethod(gray,mask,indexes)
            # gray=drawWindows(winCenters,gray)
        if(index>=11):
             cv2.waitKey()
             break

        gray=cv2.resize(gray,(int(gray.shape[1]/rate),int(gray.shape[0]/rate)))
        frame = cv2.resize(frame,(int(frame.shape[1]/rate),int(frame.shape[0]/rate)))
        mask = cv2.resize(mask,(int(mask.shape[1]/rate),int(mask.shape[0]/rate)))

        mask = cv2.applyColorMap(mask,cv2.COLORMAP_BONE)
        gray = cv2.applyColorMap(gray,cv2.COLORMAP_BONE)

        vis = numpy.concatenate((frame,mask,gray), axis=1)

        cv2.imshow('',vis)
        cv2.waitKey(durationTime)

        index+=1
    end=time.time()
    print('sss',(end-start))



if __name__=='__main__':
    main()
