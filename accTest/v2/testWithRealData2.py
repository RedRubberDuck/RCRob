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
    fileName = '../resource3/mesF20S23L_3.json'
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


def generateInputAndOutput(steer, direction, dataA):
    s = 0

    i = 0
    Mes = None
    Acc = None
    inputA = None
    timestamp = 0.0
    for dataV in dataA:
        gyro_mess_v = dataV['gyro'][2]
        acc_mess_v = dataV['accel'][1]
        vel = dataV['encoder']

        timestampCC = dataV['timestamp']
        print('Dur:', (timestampCC - timestamp))
        timestamp = timestampCC

        if vel != 0.0:
            vel = 20
        else:
            vel = 0.0
        if(i == 0):
            # oriantation = mesV
            Mes = np.matrix([[gyro_mess_v]])
            Acc = np.matrix([[acc_mess_v]])
            inputA = np.matrix([[vel/100.0*direction*1.0, steer]])
        else:
            Mes = np.concatenate((Mes, np.matrix([[gyro_mess_v]])), axis=0)
            Acc = np.concatenate((Acc, np.matrix([[acc_mess_v]])), axis=0)
            inputA = np.concatenate(
                (inputA, np.matrix([[vel/100.0*direction*1.0, steer]])), axis=0)
        i += 1

    plt.figure()
    plt.plot(Mes.T[0, :].tolist()[0])
    plt.figure()
    plt.plot(Acc.T[0, :].tolist()[0])
    plt.figure()
    plt.subplot(211)
    plt.plot(inputA.T[0, :].tolist()[0])
    plt.subplot(212)
    plt.plot(inputA.T[1, :].tolist()[0])
    plt.show()
    return Mes, Acc, inputA


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
    direction = 1.0
    mesA, AccA, inputA = generateInputAndOutput(
        np.radians(23.0), direction, data)

    rob1 = systemTest.RobotEKF(
        wheelbase=0.265, dt=0.05, std_v=0.001, std_alpha=np.radians(2))

    rob1.P = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, np.radians(10), 0.0], [0.0, 0.0, 0.0, 0.0]])
    rob1.x = np.matrix([[0.0], [0.0], [0.0], [0.0]])

    rob2 = systemTest.RobotEKF(
        wheelbase=0.265, dt=0.05, std_v=0.05, std_alpha=np.radians(1))

    rob2.P = np.matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    rob2.x = np.matrix([[0.0], [0.0], [0.0], [0.0]])

    X = rob1.x.copy()
    X_sim = rob1.x.copy()
    # #

    for Mess, Acc, U in zip(mesA, AccA, inputA):
        #     # print(U)
        rob1.predict(U.T)
        rob2.predict(U.T)
        # #     # curX = rob1.x
        # #     # # curX[2, 0] = Mess
        # #     # print('SSS', curX, Mess)
        # #      0.0131544971098
        rob1.update(Mess.T, R=np.matrix([[np.radians(2)**2]]))
        # #     # [[0.03**2*2, 0, 0], [0, 0.03**2*2, 0], [0, 0, np.radians(10)**2]]))
        X = np.concatenate((X, rob1.x.copy()), axis=1)
        X_sim = np.concatenate((X_sim, rob2.x.copy()), axis=1)
        #     index += 1

    plt.figure()
    plt.plot(X[0, :].tolist()[0], X[1, :].tolist()[0], '--')
    plt.plot(X_sim[0, :].tolist()[0], X_sim[1, :].tolist()[0], '--')
    plt.legend(['Filtered', 'Simulated'])
    plt.title("Robot Position")
    plt.figure()
    plt.plot(X[3, :].tolist()[0], 'r')
    plt.plot(mesA.T[0, :].tolist()[0], '--g')
    plt.plot(X_sim[3, :].tolist()[0], '--b')
    plt.legend(['Filtered', 'Mess', 'Simulated'])
    plt.title("Anguler velocity")
    plt.show()


if (__name__ == '__main__'):
    main()
