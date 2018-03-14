import io
import time
import threading
import picamera
import cv2
import numpy


class MyPiCamera(threading.Thread):
    picamera_obj = picamera.PiCamera()

    def __init__(self):
        MyPiCamera.picamera_obj.resolution = (1648, 1232)
        MyPiCamera.picamera_obj.framerate = 10
        self.stream = io.BytesIO()
        self.isActive = False
        self.frame = None
        self.index = 0
        super(MyPiCamera, self).__init__()

    # Read from the stream the frame and flush the stream
    def _readFromStream(self):
        self.index += 1
        print("Frame no."+str(self.index))
        start = time.time()
        self.stream.seek(0)
        buff = numpy.fromstring(
            self.stream.getvalue(), dtype=numpy.uint8)
        self.frame = cv2.imdecode(buff, 1)
        end = time.time()
        print("D", end-start)
        self.stream.seek(0)
        self.stream.truncate()

    # Run function of the thread, it reads the frame
    def run(self):
        self.index = 0
        MyPiCamera.picamera_obj.capture_sequence(
            self._open_stream(), use_video_port=True, format='bgr')

    def _open_stream(self):
        while(self.isActive):
            s = 0
            yield self.stream
            self._readFromStream()

    # Get last frame
    def getFrame(self):
        return self.frame

    # Start the thread
    def start(self):
        self.isActive = True
        super(MyPiCamera, self).start()

    # Stop the thread
    def stop(self):
        self.isActive = False


def main():
    print("Camera test")
    myCameraThread = MyPiCamera()
    myCameraThread.start()
    time.sleep(3)
    myCameraThread.stop()
    myCameraThread.join()


if __name__ == "__main__":
    main()
