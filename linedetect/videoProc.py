import cv2


class VideoReader:
    def __init__(self,file):
        self.capture = cv2.VideoCapture()
        self.file = file
    def generateFrame(self):
        self.capture.open(self.file)
        while (self.capture.isOpened()):
            ret,frame = self.capture.read()
            if ret:
                yield frame
            else:
                break
        self.capture.release()
        

