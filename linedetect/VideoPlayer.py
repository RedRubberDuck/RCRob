import numpy as np
import cv2,time,os



def preprocess2(frame,rate):
    gray = frame

    kernelsize=21
    gray = cv2.GaussianBlur(gray,(kernelsize,kernelsize),0)
    thres,mask2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    mask2 = cv2.blur(mask2,(151,151),0)
    thres,mask2 = cv2.threshold(mask2, 180, 255, cv2.THRESH_BINARY)
    mask2 = cv2.dilate(mask2,np.ones((50,50)))
    mask2 = cv2.bitwise_not(mask2)

    mask= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,191,-60)
    res = cv2.bitwise_and(mask,mask2)
    
    return res,mask,mask2


def preprocess1(frame,rate):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    
    gray=cv2.equalizeHist(gray)
    # gray = cv2.GaussianBlur(gray,(21,21),5)

    # mask = cv2.Canny(frame,100,100,L2gradient=True)

    # thresh,mask = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)

    # mask = 255-gray
    # mask= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,71,-2)
    mask= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,-20)


    # print(thresh)
    # gray = cv2.blur(gray,(11,11),0)
    # gray = cv2.blur(gray,(11,11),0)
    return mask

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
    inputFolder= os.path.realpath('../../resource/videos')
    # inputFileName='/f_big_50_4.h264'
    # inputFileName='/record19Feb/test50_7.h264'
    inputFileName='/record19Feb2/test40L_2.h264'
    # inputFileName='/newRecord/move19.h264'
    cap = cv2.VideoCapture(inputFolder+inputFileName)
    M,M_inv,newsize=getPerspectiveTransformationMatrix()

    rate=2
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)

        img = cv2.warpPerspective(gray,M,newsize)
        # img = 
        # frame = cv2.warpPerspective(frame,M,newsize)
        # img=cv2.resize(img,(int(img.shape[1]/rate),int(img.shape[0]/rate)))    
        img,mask,mask2=preprocess2(img,1)
        
        img=cv2.resize(img,(int(img.shape[1]/rate),int(img.shape[0]/rate)))
        mask=cv2.resize(mask,(int(mask.shape[1]/rate),int(mask.shape[0]/rate)))
        mask2=cv2.resize(mask2,(int(mask2.shape[1]/rate),int(mask2.shape[0]/rate)))

        resImages = np.concatenate((img,mask,mask2), axis=1)

        if(frame is not None):
            cv2.imshow('frame',resImages)
        else:
            break
        # time.sleep(0.01)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()