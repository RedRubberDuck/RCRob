import sys

sys.path.append('../Capture')
sys.path.append('../linedetect')
sys.path.append('../Control')

import myCamera
import LaneDetection



import time
import cProfile


class LaneDetecWithCamera:
    def __init__(self):
        rate = 3
        self.myCamera = myCamera.MyPiCamera(rate)
        self.laneDetectionThread = LaneDetection.LaneDetectThread(self.myCamera.getFrame, rate)

        self.myCamera.addNewEvent(
            self.laneDetectionThread, self.laneDetectionThread.newFrameEvent)

    def start(self):
        self.myCamera.start()
        self.laneDetectionThread.activate()
        self.laneDetectionThread.start()

    def stop(self):
        self.laneDetectionThread.stop()
        self.laneDetectionThread.join()
        self.myCamera.stop()
        self.myCamera.join()


def main():
    app = LaneDetecWithCamera()
    app.start()
    time.sleep(3)
    app.stop()


if __name__ == "__main__":
    cProfile.run("main()")
