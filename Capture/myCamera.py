import io
import time
import threading
import picamera
import cv2
import numpy

import ImageUtility


class MyPiCamera(threading.Thread):
    picamera_obj = picamera.PiCamera()

    def __init__(self, rate):
        self.size = (1664, 1232)
        self.rate = rate
        self.newsize = ImageUtility.calcResize(self.size, self.rate)
        MyPiCamera.picamera_obj.resolution = self.size
        MyPiCamera.picamera_obj.framerate = 15
        self.stream = io.BytesIO()
        self.isActive = False
        self.frame = None
        self.index = 0

        self.eventDic = {}
        super(MyPiCamera, self).__init__()

    # Read from the stream the frame and flush the stream
    def _readFromStream(self):

        self.stream.seek(0)
        buff = numpy.fromstring(
            self.stream.getvalue(), dtype=numpy.uint8)
        self.frame = buff.reshape((self.newsize[1], self.newsize[0], 4))
        self.index += 1
        self.stream.seek(0)
        self.stream.truncate()
        self.setEvents()

    # Run function of the thread, it reads the frame
    def run(self):
        self.index = 0
        MyPiCamera.picamera_obj.capture_sequence(
            self._open_stream(), use_video_port=True, format='bgra', resize=self.newsize)

    def _open_stream(self):
        while(self.isActive):
            s = 0
            yield self.stream
            self._readFromStream()

    # Get last frame
    def getFrame(self):
        return (self.index, self.frame)

    # Start the thread
    def start(self):
        self.isActive = True
        super(MyPiCamera, self).start()

    # Stop the thread
    def stop(self):
        self.isActive = False

    def setEvents(self):
        for key, event in self.eventDic.items():
            event.set()

    def addNewEvent(self, key, event):
        self.eventDic[key] = event


class Saver(threading.Thread):
    def __init__(self, getFrameFnc):
        super(Saver, self).__init__()
        self.name = 's'
        self.event = threading.Event()
        self.isAlive = True
        self.getFrameFnc = getFrameFnc

    def run(self):
        start = time.time()
        while(self.isAlive):
            if(self.event.wait(0.01)):
                index, frame = self.getFrameFnc()
                end = time.time()
                print("No.", index)
                print('D', end-start)
                start = end

                # cv2.imwrite("i"+str(index)+".jpg", frame)


def main():
    print("Camera test")
    myCameraThread = MyPiCamera(1)
    saver = Saver(myCameraThread.getFrame)
    myCameraThread.addNewEvent(1, saver.event)
    saver.start()
    myCameraThread.start()
    time.sleep(3)
    saver.isAlive = False
    myCameraThread.stop()
    myCameraThread.join()


if __name__ == "__main__":
    main()
