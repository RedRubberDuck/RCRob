import json
import numpy as np
from matplotlib import pyplot as plt

import systemTest

'''
# Player One
# Ragodozo varosok
# Expedocio
*0.430875709375
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
    Acc = None
    for dataV in dataA:
        gyro_mess_v = dataV['gyro'][2]
        acc_mess_v = dataV['accel'][1]

        if(i == 0):
            # oriantation = mesV
            Mes = np.matrix([[gyro_mess_v]])
            Acc = np.matrix([[acc_mess_v]])
        else:
            Mes = np.concatenate((Mes, np.matrix([[gyro_mess_v]])), axis=0)
            Acc = np.concatenate((Acc, np.matrix([[acc_mess_v]])), axis=0)
        i += 1
    plt.figure()
    plt.plot(Mes.T[0, :].tolist()[0])
    plt.figure()
    plt.plot(Acc.T[0, :].tolist()[0])
    plt.show()
    return Mes, Acc


def plotting(dataA):

    GyroA = None
    AccA = None
    CompassA = None
    FussionA = None
    i = 0
    for dataV in dataA:
        if i == 0:
            GyroA = np.matrix([dataV['gyro']])
            AccA = np.matrix([dataV['accel']])
            CompassA = np.matrix([dataV['compass']])
            FussionA = np.matrix([dataV['fusionPose']])
            # print(GyroA)
        else:
            GyroA = np.concatenate((GyroA, np.matrix([dataV['gyro']])), axis=0)
            AccA = np.concatenate((AccA, np.matrix([dataV['accel']])), axis=0)
            CompassA = np.concatenate(
                (CompassA, np.matrix([dataV['compass']])), axis=0)
            FussionA = np.concatenate(
                (FussionA, np.matrix([dataV['fusionPose']])), axis=0)
        i += 1

    plt.figure()
    plt.subplot(311)
    plt.title("Accel")
    plt.plot(AccA[:, 0].T.tolist()[0])
    plt.subplot(312)
    plt.plot(AccA[:, 1].T.tolist()[0])
    plt.subplot(313)
    plt.plot(AccA[:, 2].T.tolist()[0])
    plt.figure()

    plt.subplot(311)
    plt.title("Gyro")
    plt.plot(GyroA[:, 0].T.tolist()[0])
    plt.subplot(312)
    plt.plot(GyroA[:, 1].T.tolist()[0])
    plt.subplot(313)
    plt.plot(GyroA[:, 2].T.tolist()[0])
    plt.figure()

    plt.subplot(311)
    plt.title("Compass")
    plt.plot(CompassA[:, 0].T.tolist()[0])
    plt.subplot(312)
    plt.plot(CompassA[:, 1].T.tolist()[0])
    plt.subplot(313)
    plt.plot(CompassA[:, 2].T.tolist()[0])
    plt.figure()

    plt.subplot(311)
    plt.title("Fussion")
    plt.plot(np.degrees(FussionA[:, 0].T.tolist()[0]))
    plt.subplot(312)
    plt.plot(np.degrees(FussionA[:, 1].T.tolist()[0]))
    plt.subplot(313)
    plt.plot(np.degrees(FussionA[:, 2].T.tolist()[0]))

    plt.show()


def gyroStd(dataA):
    s = 0

    gyroA = []
    for dataV in dataA:
        gyro = dataV['gyro'][2]
        gyroA.append(gyro)

    stdGyro = np.std(gyroA)
    print('Std gyro:', stdGyro)

    # 0.0163053788053
    # 0.0131544971098
    # 0.013447431677
    # 0.0200003859432
    # 0.024478928753

    # plt.plot(gyroA)
    # plt.show()


def main():
    data = readFile()
    gyroStd(data)
    # plotting(data)

    inputA = generateInput(20, 0.0)
    mesA, AccA = generateOutput(data)

    rob1 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=0.01, std_alpha=np.radians(3))

    rob1.P = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rob1.x = np.matrix([[0.0], [0.0], [0.0], [0.0]])

    rob2 = systemTest.RobotEKF(
        wheelbase=0.26, dt=0.02, std_v=0.05, std_alpha=np.radians(1))

    rob2.P = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rob2.x = np.matrix([[0.0], [0.0], [0.0], [0.0]])

    X = rob1.x.copy()
    X_sim = rob1.x.copy()
    #

    speed = -0.2
    direction = speed/abs(speed)

    forwardSpeed = 0
    moving = False
    steer = np.radians(0.0)

    index = 0

    startIndex = 0
    endIndex = 0
    for Mess, Acc in zip(mesA, AccA):
        if not moving and Acc[0, 0]*direction > 0.2:
            forwardSpeed = speed
            moving = True
            startIndex = index
            print("Forward")
        elif moving and Acc[0, 0]*direction < -0.15:
            forwardSpeed = 0.0001
            moving = False
            endIndex = index
            print("Brake")
        U = np.matrix([[forwardSpeed+np.random.normal(loc=0.0, scale=0.01)],
                       [steer+np.random.normal(loc=0.0, scale=np.radians(3.0))]])
        # print(U)
        rob1.predict(U)
        rob2.predict(U)
    #     # curX = rob1.x
    #     # # curX[2, 0] = Mess
    #     # print('SSS', curX, Mess)
    #      0.0131544971098
        rob1.update(Mess.T, R=np.matrix([[(0.0231544971098)**2]]))
    #     # [[0.03**2*2, 0, 0], [0, 0.03**2*2, 0], [0, 0, np.radians(10)**2]]))
        X = np.concatenate((X, rob1.x.copy()), axis=1)
        X_sim = np.concatenate((X_sim, rob2.x.copy()), axis=1)
        index += 1

    print('Duration', (endIndex - startIndex)*0.02, (endIndex - startIndex))
    plt.figure()
    plt.plot(X[0, :].tolist()[0], X[1, :].tolist()[0])
    plt.plot(X_sim[0, :].tolist()[0], X_sim[1, :].tolist()[0])
    plt.legend(['Filtered', 'Simulated'])
    plt.figure()
    plt.plot(X[3, :].tolist()[0], 'r')
    plt.plot(mesA.T[0, :].tolist()[0], '--g')
    plt.plot(X_sim[3, :].tolist()[0], '--b')
    plt.legend(['Filtered', 'Mess', 'Simulated'])
    plt.show()


if (__name__ == '__main__'):
    main()
