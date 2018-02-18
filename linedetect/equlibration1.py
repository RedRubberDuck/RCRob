import cv2, time, os
from matplotlib import pyplot as plt
import numpy as np



def processFrame2(frame):
    img_size=(frame.shape[1],frame.shape[0])

    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(10,10))
    # hsv[:,:,1] = clahe.apply(hsv[:,:,1])
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])
    bgr=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print(hsv.shape)
    print(bgr.shape)


    lower_white = np.array([0,0,200], dtype=np.uint8)
    upper_white = np.array([255,20,255], dtype=np.uint8)
    # #---------------------------------------------------------------------------

    mask = cv2.inRange(hsv, lower_white, upper_white)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,gray,mask = mask)

    return (gray,mask,bgr)

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

def convertImage(img,rate):
    img = cv2.resize(img,(int(img.shape[1]/rate),int(img.shape[0]/rate)))
    # img = cv2.applyColorMap(img,cv2.COLORMAP_BONE)
    return img    



def main():
    # inputFolder='/home/nandi/Workspaces/Work/Python/opencvProject/Apps/pics/videos/'
    # inputFolder='C:\\Users\\aki5clj\\Documents\\PythonWorkspace\\Rpi\\Opencv\\LineDetection\\resource\\'
    inputFolder= os.path.realpath('../../resource/videos')
    inputFileName='/newRecord/move14.h264'
    print(inputFolder+inputFileName)
    frameGenerator=videoRead(inputFolder+inputFileName)
    start=time.time()
    rate=4
    index=0

    M,M_inv,newsize=getPerspectiveTransformationMatrix()
    print(newsize)
    
    lines=[]
    index=0
    nrSlices=15
    
    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(40,40))
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(80,80))

    
    for frame,durationTime in frameGenerator:
        frame = cv2.warpPerspective(frame,M,newsize)
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask1 = cv2.inRange(gray, 200 ,255)
        mask1F = cv2.GaussianBlur(mask1,(21,21),0)
        mask1FR = cv2.inRange(mask1F, 50 ,220)
        mask1FRE = cv2.GaussianBlur(mask1FR,(61,61),0)
        mask1FRER = cv2.inRange(mask1FRE, 100 ,255)
        
        
        
        
        gray = convertImage(gray,rate)
        mask1 = convertImage(mask1,rate)
        mask1F = convertImage(mask1F,rate)
        mask1FR = convertImage(mask1FR,rate)
        mask1FRE = convertImage(mask1FRE,rate)
        mask1FRER = convertImage(mask1FRER,rate)


        

    
        
        ver1 = np.concatenate((gray,mask1,mask1F), axis=1)
        ver2 = np.concatenate((mask1FR,mask1FRE,mask1FRER), axis=1)
        # ver3 = np.concatenate((grayFR,mask1FR,mask2FR), axis=1)
        # ver4 = np.concatenate((grayRes,mask1Res,mask2Res), axis=1)
        # print(ver2.shape,ver3.shape)
        # ,ver2,ver3

        # allImg=ver1
        allImg = np.concatenate((ver1,ver2), axis=0)
        cv2.imshow('',allImg)
        # cv2.waitKey(durationTime)
        cv2.waitKey()

        index+=1
    end=time.time()
    print('sss',(end-start))



if __name__=='__main__':
    main()