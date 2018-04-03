import KalmanFilter as myKalmanFilter
import json
import numpy as np
from matplotlib import pyplot as plt


def readFile():
    fileName = '../resource/data3Out.json'
    fileIn = open(fileName, 'r')
    data = json.load(fileIn)
    # print(data)
    fileIn.close()
    return data


def generateInput(data):
    inputVal = []
    steer = np.radians(0)
    i = 0
    Gyro_A = []
    for dataV in data:
        if i > 5+42:
            forwardAccel_v = -1000 * dataV['accel'][1]
            print(forwardAccel_v)
            Gyro_A .append(np.radians(dataV['gyro'][2]))
            inputVal.append(np.matrix([[forwardAccel_v], [steer]]))
        if i > 160+20+42:
            break
        i += 1

    return inputVal, Gyro_A


def plotting(dataA):

    accX = []
    accY = []
    accZ = []
    gyroX = []
    gyroY = []
    gyroZ = []

    i = 0
    for data in dataA:
        if i > 5+42:
            accX.append(data['accel'][0]*1000)
            accY.append(data['accel'][1]*-1000)
            accZ.append(data['accel'][2]*1000)

            gyroX.append(np.radians(data['gyro'][0]))
            gyroY.append(np.radians(data['gyro'][1]))
            gyroZ.append(np.radians(data['gyro'][2]))
        if i > 150+10+42:
            break
        i += 1

    plt.figure()
    plt.subplot(311)
    plt.plot(accX)
    plt.subplot(312)
    plt.plot(accY)
    plt.subplot(313)
    plt.plot(accZ)

    plt.figure()
    plt.subplot(311)
    plt.plot(gyroX)
    plt.subplot(312)
    plt.plot(gyroY)
    plt.subplot(313)
    plt.plot(gyroZ)
    plt.show()


def main():
    data = readFile()
    plotting(data)
    inputValArray, Gyro_A = generateInput(data)
    timestep = 0.02
    wheelbase = 26

    rob1 = myKalmanFilter.RobotEKF(dt=timestep, wheelbase=wheelbase,
                                   std_acc=50.0, std_steer=np.radians(2))

    X = rob1.x
    # inputAcc = []
    i = 0
    for inputV, gyroV in zip(inputValArray, Gyro_A):

        rob1.predict(inputV)
        # if i < 2:
        #     rob1.update(np.array([[0.0], [gyroV]]), R=np.matrix([[1**2, 0], [0, 0.0001**2]]))
        # elif i < 148:
        #     rob1.update(np.array([[-20.0], [gyroV]]), R=np.matrix([[5**2, 0], [0, 0.0001**2]]))
        # else:
        #     rob1.update(np.array([[0.0], [gyroV]]), R=np.matrix([[1**2, 0], [0, 0.0001**2]]))
        X_temp = rob1.x
        X = np.concatenate((X, rob1.x), axis=1)
        i += 1
    plt.figure()
    plt.title('Velocity')
    plt.plot(X[2, :].tolist()[0], '--r')

    plt.figure()
    plt.title('Oriantation')
    plt.plot(X[3, :].tolist()[0], '--r')

    plt.figure()
    plt.title('Angular velocity')
    plt.plot(X[4, :].tolist()[0], '--r')

    plt.figure()
    plt.plot(X[0, :].tolist()[0], X[1, :].tolist()[0], '--r')
    plt.show()


if __name__ == '__main__':
    main()
