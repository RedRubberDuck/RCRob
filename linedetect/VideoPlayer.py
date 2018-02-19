import numpy as np
import cv2,time,os

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

inputFolder= os.path.realpath('../../resource/videos')
# inputFileName='/f_big_50_9.h264'
# inputFileName='/record19Feb/test50_8.h264'
inputFileName='/record19Feb2/test50_1.h264'
cap = cv2.VideoCapture(inputFolder+inputFileName)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
M,M_inv,newsize=getPerspectiveTransformationMatrix()


while(cap.isOpened()):
    ret, frame = cap.read()
    # frame = cv2.warpPerspective(frame,M,newsize)
    gray=frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5),0)
    
    gray=clahe.apply(gray)
    gray = cv2.blur(gray,(11,11),0)
    # gray = cv2.blur(gray,(11,11),0)
    gray=cv2.resize(gray,(int(gray.shape[1]/2),int(gray.shape[0]/2)))

    if(frame is not None):
        cv2.imshow('frame',gray)
    else:
        break
    time.sleep(0.033)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()