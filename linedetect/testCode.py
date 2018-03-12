import frameProcessor, videoProc, drawFunction,postprocess, ImageTransformation, SlicingMethod, WindowSlidingFnc
import cv2



rate=1
persTransformation,pxpcm = ImageTransformation.ImagePerspectiveTransformation.getPerspectiveTransformation3(rate)
pxpcm = pxpcm
persTransformation = persTransformation
birdviewImage_size = persTransformation.size
frameFilter = frameProcessor.frameFilter.FrameLineSimpleFilter(persTransformation)

# img = cv2.imread('/home/nandi/Workspaces/git/RCRob/C++/img.jpg')
cap = cv2.VideoCapture()
cap.open('/home/nandi/Workspaces/git/resource/videos/martie2/test12.h264')


# for i in range(10): 
ret,frame = cap.read()


birdview_gray, birdview_mask = frameFilter.apply2(frame)

cv2.imwrite("/home/nandi/Workspaces/git/RCRob/C++/img1.jpg",birdview_mask)

cv2.imshow("",birdview_mask*255)
cv2.waitKey()
