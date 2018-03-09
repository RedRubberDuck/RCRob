import cv2
import io

class VideoReader:
    def __init__(self,file):
        self.capture = cv2.VideoCapture()
        self.file = file
        
   

    def readFrame(self):
        self.capture.open(self.file)
        while (self.capture.isOpened()):
            ret,frame = self.capture.read()
            if ret:
                yield frame
            else:
                break
        self.capture.release()
        
class VideoReaderWithResize(VideoReader):
    def __init__(self,file,rate):
        super(VideoReaderWithResize,self).__init__(file)
        self.rate  = rate
        self.stream = None
    def readFrame(self):
        for frame in super(VideoReaderWithResize,self).readFrame():
            newSize = frame.shape[1]//self.rate,frame.shape[0]//self.rate
            resizedFrame = cv2.resize(frame,newSize)
            yield resizedFrame