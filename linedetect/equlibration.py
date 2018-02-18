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
    img = cv2.applyColorMap(img,cv2.COLORMAP_BONE)
    return img    



def main():
    # inputFolder='/home/nandi/Workspaces/Work/Python/opencvProject/Apps/pics/videos/'
    # inputFolder='C:\\Users\\aki5clj\\Documents\\PythonWorkspace\\Rpi\\Opencv\\LineDetection\\resource\\'
    inputFolder= os.path.realpath('../../resource/videos')
    inputFileName='/newRecord/move1.h264'
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
    
    limit=3.0
    limit=60.0
    # clahe1 = cv2.createCLAHE(clipLimit=limit, tileGridSize=(4,4))
    clahe1 = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    # clahe1 = cv2.createCLAHE()
    # clahe2 = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
    # clahe3 = cv2.createCLAHE(clipLimit=limit, tileGridSize=(10,10))
    # clahe4 = cv2.createCLAHE(clipLimit=limit, tileGridSize=(12,12))
    # clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(128,128))


    
    for frame,durationTime in frameGenerator:
        frame = cv2.warpPerspective(frame,M,newsize)
        
        # frame = cv2.GaussianBlur(frame,(21,21),0)
        
        edges = cv2.Canny(frame,21,100)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        # hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])
        hsv[:,:,2] = clahe1.apply(hsv[:,:,2])
        hsv[:,:,2] = cv2.GaussianBlur(hsv[:,:,2],(21,21),0)
        lower_white = np.array([0,0,230], dtype=np.uint8)
        upper_white = np.array([255,50,255], dtype=np.uint8)
        # #---------------------------------------------------------------------------

        bin1 = cv2.inRange(hsv, lower_white, upper_white)
        bin2 = cv2.inRange(hsv1, lower_white, upper_white)
        grayE1 = hsv[:,:,2]



        
        gray  = convertImage(gray,rate)
        gray1 = convertImage(grayE1,rate)
        bin1   = convertImage(bin1,rate)
        bin2  = convertImage(bin2,rate)
        edges  = convertImage(edges,rate)

        
        # allImg=ver1
        allImg = np.concatenate((gray,gray1,bin2,bin1,edges), axis=1)
        cv2.imshow('',allImg)
        # cv2.waitKey(durationTime)
        cv2.waitKey()

        index+=1
    end=time.time()
    print('sss',(end-start))



if __name__=='__main__':
    main()