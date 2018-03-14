import io
import time
import threading
import picamera
import cv2
import numpy

# Create a pool of image processors
done = False
lock = threading.Lock()
eventR = threading.Event()
eventW = threading.Event()


class ImageSaver:
    def __init__(self):
        self.index = 0
        self.lock = threading.Lock()

    def save(self, frame):
        with self.lock:
            index = self.index
            self.index += 1
        cv2.imwrite("img"+str(index)+".jpg", frame)
        print("Saved img"+str(index)+".jpg")


class ImageProcessor(threading.Thread):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.saver = ImageSaver()
        self.isActive = True

    def run(self):
        while(self.isActive):
            eventR.wait()
            eventR.clear()
            try:

                self.stream.seek(0)
                buff = numpy.fromstring(
                    self.stream.getvalue(), dtype=numpy.uint8)
                frame = cv2.imdecode(buff, 1)
            finally:
                self.stream.seek(0)
                self.stream.truncate()
            eventW.set()
                self.saver.save(frame)


class ImageProcessorManager:
    # Constructor
    #  @param  nrImageProcessor             The number of ImageProcessor objects
    def __init__(self, nrImageProcessor, processFnc):
        self.name = "ImageProcessorManager"
        self.processor = ImageProcessor()
        self.processor.start()

        self.poolLock = threading.Lock()
        self.isRunning = True

    # Start the thread running
    def start(self):
        if not self.isRunning:
            self.isRunning = True
            # super(ImageProcessorManager,self).start()
    # Stop the thread running

    def stop(self):
        self.processor.isActive = False
        self.processor.join()
        self.isRunning = False

    # Run function
    def run(self):
        startTime = time.time()
        while(self.isRunning):
            # Block the pool accessing

            yield self.processor.stream
            eventR.set()
            eventW.wait()
            eventW.clear()


def main():
    imageSaver = ImageSaver()
    manager = ImageProcessorManager(30, imageSaver.save)
    with picamera.PiCamera() as camera:
        camera.resolution = (1648, 1232)
        camera.framerate = 30

        try:
            camera.capture_sequence(manager.run(), use_video_port=True)
        except KeyboardInterrupt:
            print("KeyboardInterrupt_Exit")
            pass
        finally:
            manager.stop()


if __name__ == '__main__':
    main()
