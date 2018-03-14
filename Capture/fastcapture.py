import io
import time
import threading
import picamera
import cv2
import numpy

# Create a pool of image processors
done = False
lock = threading.Lock()
pool = []


class ImageProcessor(threading.Thread):
    def __init__(self, addToPoolFunc, frameCollecting, processFnc):
        super(ImageProcessor, self).__init__()
        self.name = "ImageProcessor"
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.addToPoolFunc = addToPoolFunc
        self.frameCollecting = frameCollecting
        self.processFnc = processFnc
        self.start()
        self.frameID = None

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
                    frame = cv2.imdecode(buff, 1)
                    self.processFnc(frame)
                    self.frameCollecting(self.frameID, frame)

                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the pool
                    if(not self.terminated):
                        self.addToPoolFunc(self)


class ImageProcessorManager:
    # Constructor
    #  @param  nrImageProcessor             The number of ImageProcessor objects
    def __init__(self, nrImageProcessor, processFnc):
        self.name = "ImageProcessorManager"
        self.pool = [ImageProcessor(self.addToPool, self.frameCollecting, processFnc)
                     for i in range(nrImageProcessor)]
        self.poolLock = threading.Lock()
        self.isRunning = True
        self.frameMap = {}

    def addToPool(self, obj):
        with self.poolLock:
            self.pool.append(obj)

    def frameCollecting(self, index, frame):
        self.frameMap[index] = frame
        if len(self.frameMap) > 3:
            frame = self.frameMap.popitem()
            print('Deleted', frame[0])
            if frame[1] is None:
                print("Process !!!")
                # Start the thread running

    def start(self):
        if not self.isRunning:
            self.isRunning = True
            # super(ImageProcessorManager,self).start()
    # Stop the thread running

    def stop(self):
        self.isRunning = False

    def stopAllImageProcessor(self):
        threads = threading.enumerate()
        for activeThread in threads:
            if activeThread.getName() == "ImageProcessor":
                activeThread.stop()
                activeThread.join()

    # Run function
    def run(self):
        index = 0
        startTime = time.time()
        while(self.isRunning):
            # Block the pool accessing
            with self.poolLock:
                if len(self.pool) != 0:
                    processor = self.pool.pop()
                else:

                    processor = None

            if processor:
                yield processor.stream
                processor.frameID = index
                self.frameMap[index] = None
                processor.event.set()
                endtime = time.time()
                print("Duration:", (endtime-startTime))
                index += 1
                if index > 10:
                    break
                startTime = endtime
            else:
                # When the pool is starved, wait a while for it to refill
                time.sleep(0.001)


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


def main():
    imageSaver = ImageSaver()
    manager = ImageProcessorManager(30, imageSaver.save)
    with picamera.PiCamera() as camera:
        camera.resolution = (1648, 1232)
        camera.framerate = 10
        camera.start_preview()

        try:
            camera.capture_sequence(
                manager.run(), format='bgr', use_video_port=True)
        except KeyboardInterrupt:
            print("KeyboardInterrupt_Exit")
            pass
        finally:
            manager.stop()
            manager.stopAllImageProcessor()


if __name__ == '__main__':
    main()
