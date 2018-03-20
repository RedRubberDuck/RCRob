#!/usr/bin/env python3

import cv2

import videoProc
import drawFunction
import LaneDetection
import os
import cv2
import cv2.plot
import time
import numpy as np

from matplotlib import pyplot as plt
import BezierCurve

import cProfile


def main():

    print("Test10.py -Main-")
    # source folder
    inputFolder = os.path.realpath('../../resource/videos')
    # source file

    # inputFileName='/newRecord/move4.h264'
    inputFileName = '/martie2/test12.h264'

    print('Processing:', inputFolder+inputFileName)
    rate = 2
    # Video frame reader object
    videoReader = videoProc.VideoReaderWithResize(
        inputFolder+inputFileName, rate)
    frameRate = 30.0
    frameDuration = 1.0/frameRate

    # with PyCallGraph(output=graphviz):
    start = time.time()

    laneDetec = LaneDetection.LaneDetector(rate)

    for frame in videoReader.readFrame():
        laneDetec.frameProcess(frame)


if __name__ == '__main__':
    cProfile.run('main()')
#
    # main()
