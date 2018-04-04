import json
import numpy as np
from matplotlib import pyplot as plt

import systemTest

'''
Player One
Ragodozo varosok
Expedocio
'''


def readFile():
    fileName = '../resource2/dataF20.json'
    fileIn = open(fileName, 'r')
    data = json.load(fileIn)
    # print(data)
    fileIn.close()
    return data


def generateInput(vel, steer):
    inputA = np.matrix([[0.0, 0.0]])

    for i in range(500):
        inputV = np.matrix([[vel, steer]])
        inputA = np.concatenate((inputA, inputV), axis=0)

    return inputA


def generateOutput(dataA):
    s = 0

    i = 0
    Mes = None
    for dataV in dataA:
        mesV = dataV['fusionPose'][2]
        if(i == 0):
            oriantation = mesV
            Mes = np.matrix([[mesV]])
            s = 0
        else:
            Mes = np.concatenate((Mes, np.matrix([[mesV]])), axis=0)
            s = 0
        i += 1
    plt.plot(Mes.T[0, :].tolist()[0])
    plt.show()
    return Mes, oriantation


def plotting(dataA):
    s = 0.0


def main():
    data = readFile()
    # plotting(data)
    inputA = generateInput(20, 0.0)
    mesA, oriantation = generateOutput(data)

    rob1 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=0.01, std_alpha=np.radians(0.9))

    rob1.P = np.matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rob1.x = np.matrix([[0.0], [0.0], [oriantation]])
    X = rob1.x.copy()

    for U, Mess in zip(inputA, mesA):
        rob1.predict(U.T)
        # curX = rob1.x
        # # curX[2, 0] = Mess
        # print('SSS', curX, Mess)
        # rob1.update(Mess.T, R=np.matrix([[np.radians(10)**2]]))
        # [[0.03**2*2, 0, 0], [0, 0.03**2*2, 0], [0, 0, np.radians(10)**2]]))
        X = np.concatenate((X, rob1.x.copy()), axis=1)

    plt.figure()
    plt.plot(X[0, :].tolist()[0], X[1, :].tolist()[0])
    # plt.plot(Outputs[0, :].tolist()[0], Outputs[1, :].tolist()[0])
    plt.figure()
    plt.plot(X[2, :].tolist()[0])
    plt.plot(Mess.T[0, :].tolist()[0])
    plt.show()


if (__name__ == '__main__'):
    main()
