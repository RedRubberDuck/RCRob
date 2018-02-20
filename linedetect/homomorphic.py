import numpy as np
import cv2,time,os
import VideoPlayer
from  matplotlib import pyplot as plt

def main():
    inputFolder= os.path.realpath('../../resource/videos')
    inputFileName='/f_big_50_4.h264'
    # inputFileName='/record19Feb/test50_5.h264'
    inputFileName='/record19Feb2/test50L_1.h264'
    # inputFileName='/move17.h264'
    # inputFileName='/newRecord/move1.h264'
    # inputFileName='/record20Feb/test5_1.h264'
    cap = cv2.VideoCapture(inputFolder+inputFileName)
    M,M_inv,newsize=VideoPlayer.getPerspectiveTransformationMatrix()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(46,46))
    while(cap.isOpened()):
        ret, frame = cap.read()
        # frame = cv2.warpPerspective(frame,M,newsize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
    
        
        size =(int(frame.shape[1]/2),int(frame.shape[0]/2))


        dft = cv2.dft(np.float32(gray),flags = cv2.DFT_COMPLEX_OUTPUT)
        # print(dft)
        
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum_dft = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

        
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        rows, cols = gray.shape
        crow,ccol = int(rows/2) , int(cols/2)
        mask = np.ones((rows,cols,2),np.uint8)
        filterR=400
        print(int(rows/filterR))
        mask[crow-int(rows/filterR):crow+int(rows/filterR), ccol-int(cols/filterR):ccol+int(cols/filterR)] = 0

        fshift = dft_shift*mask
        magnitude_spectrum_fshift = 20*np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))


        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,0])

        # print(img_back)

        frame  = cv2.resize(frame,size)
        gray  = cv2.resize(img_back,size)
        resFinal = gray

        plt.figure()
        plt.subplot(311)
        plt.imshow(img_back, cmap = 'gray')
        plt.subplot(312)
        plt.imshow(magnitude_spectrum_dft, cmap = 'gray')
        plt.subplot(313)
        plt.imshow(magnitude_spectrum_fshift, cmap = 'gray')
        plt.show()

        # if(frame is not None):
        #     # cv2.imshow('frame',resFinal)
        # else:
        #     break
        # time.sleep(0.01)
        # if cv2.waitKey(33) & 0xFF == ord('q'):
            # break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()