import io
import time
import threading
import picamera
import cv2
import numpy



class ImageProcessor(threading.Thread):
    def __init__(self, addToPoolFunc, frameCollecting, processFnc,imageSize):
        super(ImageProcessor, self).__init__()
        self.name = "ImageProcessor"
        self.stream = io.BytesIO()
        self.event = threading.Event()

        
        self.addToPoolFunc = addToPoolFunc
        self.frameCollecting = frameCollecting
        self.processFnc = processFnc

        self.terminated = False
        self.frameID = None
        self.imageSize = imageSize

        self.start()
        

    def stop(self):
        self.terminated = True

    def run(self):
        # This method runs in a separate thread
        global done
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(0.01):
                try:
                    self.stream.seek(0)
                    buff = numpy.fromstring(
                        self.stream.getvalue(), dtype=numpy.uint8)
                    frame = buff.reshape((self.imageSize[1],self.imageSize[0],4))
                    # frame = cv2.imdecode(buff, 1)
                    if (self.processFnc is not None ):
                        frame = self.processFnc(self.frameID,frame) 
                    print('Processed frame',self.frameID)
                    self.frameCollecting(self.frameID, frame)

                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the pool
                    if(not self.terminated):
                        self.addToPoolFunc(self)