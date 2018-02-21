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

    return kernel 

rate = 4

class LineHomomorphic:
    def __init__(self,img_size,rate):
        optimalNrows = cv2.getOptimalDFTSize(int(img_size[0]/rate))
        optimalNcols = cv2.getOptimalDFTSize(int(img_size[1]/rate))
        self.img_size = (optimalNrows,optimalNcols)
        temp = gkern(l = (optimalNcols,optimalNrows),sig =0.9)
        plt.imshow(temp)
        plt.show()
        temp = 1 - temp
        normTemp = temp 
        # / np.sum(temp)
        self.kernel_highPass = np.zeros((optimalNrows,optimalNcols,2))
        self.kernel_highPass [:,:,0] = normTemp
        self.kernel_highPass [:,:,1] = normTemp

    def filterImage(self,frame):
        if frame.shape != self.img_size:
            return None
        
        dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift*self.kernel_highPass
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift,flags= cv2.DFT_SCALE)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        
        # img=np.uint8(img_back)
        return img_back

    
    
    def border(self,frame):
        if (frame.shape[0] <= self.img_size[0] and frame.shape[1] <= self.img_size[1]):
            img_with_border = np.zeros(self.img_size)
            img_with_border[:frame.shape[0],:frame.shape[1]]=frame
            return img_with_border
        else:
            return None


def main():
    inputFolder= os.path.realpath('../../resource/videos')
    # inputFileName='/f_big_50_3.h264'
    # inputFileName='/record19Feb/test50_7.h264'
    # inputFileName='/record19Feb2/test50L_1.h264'
    # inputFileName='/move1.h264'
    inputFileName='/newRecord/move1.h264'
    # inputFileName='/record20Feb/test5_1.h264'
    cap = cv2.VideoCapture(inputFolder+inputFileName)
    M,M_inv,newsize=VideoPlayer.getPerspectiveTransformationMatrix()
    # clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8,8))

    rate=4
    size=(int(1232/rate),int(1648/rate))
    original_size=(1232,1648)
    print(size)
    
    gg = LineHomomorphic(original_size,4)


    optimalNrows = cv2.getOptimalDFTSize(size[0])
    optimalNcols = cv2.getOptimalDFTSize(size[1])

    kernel_highPass=gkern(l = (optimalNcols,optimalNrows),sig =6)
    kernel_highPass = np.max(kernel_highPass) - kernel_highPass
    kernel_highPass = kernel_highPass/np.sum(kernel_highPass)

    kernel_LowPass=gkern(l = (optimalNcols,optimalNrows),sig = 160/rate)

    kernel=kernel_highPass
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.warpPerspective(frame,M,newsize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(size[1],size[0]))
        # gray = clahe.apply(gray)
        start_time=time.time()


        img_border = gg.border(gray)

        img_border = np.log(img_border + 1)

        img = gg.filterImage(img_border)

        img = np.uint8(np.exp(img) - 1)

        print(np.min(img))

        # I = np.float32(gray)
        # I = np.log(I + 1)
        
        # sizeK= (2*gray.shape[1]+1,2*gray.shape[0]+1)

        # kernel = gkern(sizeK,10)
        # kernel = 1 - kernel
        # # kernel = kernel / np.sum(kernel)

        # kernel = np.fft.fftshift(kernel)
        # kernel = kernel / np.sum(kernel)

        # If =np.fft.fft2(I,sizeK)

        # # print(If.shape)
        # IfR = np.dot(kernel,If)
        # iIf = np.fft.ifft2(IfR)
        # Ifreal = np.real(iIf)
        

        # Iout = Ifreal [0:size[0],0:size[1]]
        # Ihmf =np.uint8( np.exp(Iout) - 1)
        # print(np.max(Ihmf),np.min(Ihmf))
        






        img_small = img[:size[0],:size[1]]
        # img_small = gray
        # mask = img[:size[0],:size[1]]

        mask= cv2.adaptiveThreshold(img_small,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-1)
        # mask= cv2.adaptiveThreshold(img_small,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,201,-10)

        # thres,mask = cv2.threshold(Ihmf,50,255,cv2.THRESH_BINARY)


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

        resFinal = np.concatenate((img_small,mask), axis=1)

        print('Duration',Duration)

        if(frame is not None):
            cv2.imshow('frame',resFinal)
        else:
            break
        time.sleep(0.01)
        if cv2.waitKey(int(66-Duration*1000)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()