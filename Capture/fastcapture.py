import io
import time
import threading
import picamera
import cv2
import numpy


import ImageWorker, ImageSave, ImageUtility



size = (1664,1232)
rate = 4
resize = ImageUtility.calcResize(size,rate)



class ImageProcessorManager:
    # Constructor
    #  @param  nrImageProcessor             The number of ImageProcessor objects
    def __init__(self, nrImageProcessor, processFnc):
        self.name = "ImageProcessorManager"
        self.pool = [ImageWorker.ImageProcessor(self.addToPool, self.frameCollecting, processFnc,resize)
                     for i in range(nrImageProcessor)]
        self.poolLock = threading.Lock()
        self.isRunning = True
        self.frameMap = {}

    def addToPool(self, obj):
        with self.poolLock:
            self.pool.append(obj)

    def frameCollecting(self, index, frame):
        self.frameMap[index] = frame
        if len(self.frameMap) > 4:
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
        s = 0 
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
                print("Add no.",index)
                
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




def main():
    imageSaver = ImageSave.ImageSaver()
    manager = ImageProcessorManager(4, imageSaver.save)
    with picamera.PiCamera() as camera:
        camera.resolution = size
        camera.framerate = 30
        camera.start_preview()

       

        try:
            camera.capture_sequence(
                manager.run(), format='bgra', use_video_port=True ,resize = resize)
        except KeyboardInterrupt:
            print("KeyboardInterrupt_Exit")
            pass
        finally:
            manager.stop()
            manager.stopAllImageProcessor()


if __name__ == '__main__':
    main()
