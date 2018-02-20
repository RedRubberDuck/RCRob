import numpy as np
import cv2,time,os



def preprocess2(frame,rate):
    gray = frame

    kernelsize=31
    gray = cv2.GaussianBlur(gray,(kernelsize,kernelsize),0)
    thres,mask2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    mask2 = cv2.blur(mask2,(151,151),0)
    thres,mask2 = cv2.threshold(mask2, 180, 255, cv2.THRESH_BINARY)
    mask2 = cv2.dilate(mask2,np.ones((50,50)))
    mask2 = cv2.bitwise_not(mask2)
    # ADAPTIVE_THRESH_GAUSSIAN_C
    mask= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,61,-35)
    res = cv2.bitwise_and(mask,mask2)
    
    return res,mask,mask2

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
    inputFileName='/record19Feb/test50_5.h264'
    # inputFileName='/record19Feb2/test50L_6.h264'
    # inputFileName='/move17.h264'
    # inputFileName='/newRecord/move1.h264'
    # inputFileName='/record20Feb/test5_1.h264'
    cap = cv2.VideoCapture(inputFolder+inputFileName)
    M,M_inv,newsize=getPerspectiveTransformationMatrix()

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(46,46))

    rate=2
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray=cv2.equalizeHist(gray)
        # gray = clahe.apply(gray)

        img = cv2.warpPerspective(gray,M,newsize)
        # img=cv2.resize(gray,(int(gray.shape[1]/2),int(gray.shape[0]/2)))
        img = clahe.apply(img)
        # img = 
        # frame = cv2.warpPerspective(frame,M,newsize)
        # img=cv2.resize(img,(int(img.shape[1]/rate),int(img.shape[0]/rate)))    
        img1,mask,mask2=preprocess2(img,1)

        
        
        
        
        img=cv2.resize(img,(int(img.shape[1]/rate),int(img.shape[0]/rate)))
        mask=cv2.resize(mask,(int(mask.shape[1]/rate),int(mask.shape[0]/rate)))
        img1=cv2.resize(img1,(int(img1.shape[1]/rate),int(img1.shape[0]/rate)))

        resImagesGray = np.concatenate((img,mask,img1), axis=1)
        resImagesGrayMap = cv2.applyColorMap(resImagesGray, cv2.COLORMAP_BONE)

        frame  = cv2.resize(frame,(resImagesGrayMap.shape[1],resImagesGrayMap.shape[0]))

        resFinal = np.concatenate((frame,resImagesGrayMap), axis=0)


        if(frame is not None):
            cv2.imshow('frame',resFinal)
        else:
            break
        # time.sleep(0.01)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()