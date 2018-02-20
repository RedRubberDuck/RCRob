import numpy as np
import cv2,time,os
import VideoPlayer
from  matplotlib import pyplot as plt


def gkern(l=(5,5), sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ay = np.arange(-l[0] // 2 + 1., l[0] // 2 + 1.)
    ax = np.arange(-l[1] // 2 + 1., l[1] // 2 + 1.)
    yy, xx = np.meshgrid(ay, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)


def main():
    inputFolder= os.path.realpath('../../resource/videos')
    # inputFileName='/f_big_50_4.h264'
    # inputFileName='/record19Feb/test50_5.h264'
    # inputFileName='/record19Feb2/test50L_1.h264'
    # inputFileName='/move17.h264'
    inputFileName='/newRecord/move1.h264'
    # inputFileName='/record20Feb/test5_1.h264'
    cap = cv2.VideoCapture(inputFolder+inputFileName)
    M,M_inv,newsize=VideoPlayer.getPerspectiveTransformationMatrix()
    # clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8,8))

    rate=3
    size=(int(1232/rate),int(1648/rate))
    
    
    optimalNrows = cv2.getOptimalDFTSize(size[0])
    optimalNcols = cv2.getOptimalDFTSize(size[1])

    kernel=gkern(l = (optimalNcols,optimalNrows),sig = 5)
    kernel = np.max(kernel) - kernel


    while(cap.isOpened()):
        ret, frame = cap.read()
        # frame = cv2.warpPerspective(frame,M,newsize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(size[1],size[0]))
        # gray = clahe.apply(gray)
        start_time=time.time()

        nimg = np.zeros((optimalNrows,optimalNcols))
        # print(nimg.shape)
        # print(gray.shape)
        nimg[:size[0],:size[1]] = gray

        dft = cv2.dft(np.float32(nimg),flags = cv2.DFT_COMPLEX_OUTPUT)
        
        dft_shift = np.fft.fftshift(dft)

        fshift = dft_shift
        fshift[:,:,0] = dft_shift[:,:,0]*kernel
        fshift[:,:,1] = dft_shift[:,:,1]*kernel
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        img=np.uint8(img_back/np.max(img_back)*255)

        img_small = img[:size[0],:size[1]]
        mask= cv2.adaptiveThreshold(img_small,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,-20)

        # thres,mask = cv2.threshold(img,50,255,cv2.THRESH_BINARY)


        # frame  = cv2.resize(frame,size)
        # gray  = cv2.resize(img_back,size)
        # resFinal = gray

        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(gray, cmap = 'gray')
        # plt.subplot(132)
        # plt.imshow(img, cmap = 'gray')
        # plt.subplot(133)
        # plt.imshow(mask,  cmap = 'gray')
        # # plt.subplot(154)
        # # plt.imshow(magnitude_spectrum_dft, cmap = 'gray')
        # # plt.subplot(155)
        # # plt.imshow(magnitude_spectrum_fshift, cmap = 'gray')
        # plt.show()
        end_time=time.time()
        Duration=end_time-start_time
        print('Duration',Duration)

        if(frame is not None):
            cv2.imshow('frame',mask)
        else:
            break
        time.sleep(0.01)
        if cv2.waitKey(int(66-Duration*1000)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()