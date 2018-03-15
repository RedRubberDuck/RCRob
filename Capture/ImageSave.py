import io
import time
import picamera
import cv2
import numpy


class ImageSaver:
    # def __init__(self):
        # self.index = 0
        # self.lock = threading.Lock()

    def save(self, index,frame):
        # with self.lock:
        #     index = self.index
        #     self.index += 1
        cv2.imwrite("img"+str(index)+".jpg", frame)
        print("Saved img"+str(index)+".jpg")
        return frame