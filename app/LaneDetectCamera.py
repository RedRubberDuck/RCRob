import sys

sys.path.append('../Capture')
sys.path.append('../linedetect')
sys.path.append('../Control')

import myCamera
import LaneDetection
import SerialHandler
import movingScript


import time
import cProfile
import threading


class LaneDetecWithCamera:
    def __init__(self):
        rate = 3
        self.myCamera = myCamera.MyPiCamera(rate)
        self.laneDetectionThread = LaneDetection.LaneDetectThread(
            self.myCamera.getFrame, rate)

        self.myCamera.addNewEvent(
            self.laneDetectionThread, self.laneDetectionThread.newFrameEvent)

        self.serialHandler = SerialHandler.SerialHandler()
        self.controlThread = movingScript.movingStateClass(
            self.serialHandler, self.laneDetectionThread.getDistanceFromOptimelLine, self.laneDetectionThread.getFrameId)

    def start(self):
        self.serialHandler.startReadThread()
        self.myCamera.start()
        self.laneDetectionThread.activate()
        self.laneDetectionThread.start()
        self.controlThread.start()

    def stop(self):
        self.controlThread.stop()
        self.controlThread.join()
        
        # Close the lane detection thread
        self.laneDetectionThread.stop()
        self.laneDetectionThread.join()
        # Close the camera capture thread
        self.myCamera.stop()
        self.myCamera.join()
        # Close serial comunication
        self.serialHandler.close()
        


def main():
    app = LaneDetecWithCamera()
    app.start()
    time.sleep(3)
    app.stop()


if __name__ == "__main__":
    cProfile.run("main()")
