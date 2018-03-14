import io
import time
import threading
import picamera
import cv2
import numpy


class ImageProcessor(threading.Thread):
    def __init__(self, addToPoolFunc, processFnc):
        super(ImageProcessor, self).__init__()
        self.name = "ImageProcessor"
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.addToPoolFunc = addToPoolFunc
        self.processFnc = processFnc
        self.start()

    def stop(self):
        self.terminated = True

    def run(self):
        # This method runs in a separate thread
        global done
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(0.1):
                try:
                    self.stream.seek(0)
                    buff = numpy.fromstring(
                        self.stream.getvalue(), dtype=numpy.uint8)
                    frame = cv2.imdecode(buff, 1)
                    self.processFnc(frame)
                    # Read the image and do some processing on it
                    # Image.open(self.stream)
                    #...
                    #...
                    # Set done to True if you want the script to terminate
                    # at some point
                    # done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the pool
                    if(not self.terminated):
                        self.addToPoolFunc(self)
