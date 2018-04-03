import numpy as np
import math
from matplotlib import pyplot as plt
import testhelp
import plotHelp
from SystemModel import *
from KalmanFilter import *


def testSecvenceGenerate(timestep):
    alpha_a, accel_a, vel_a = testhelp.testSemnalGenerate(timestep)
    alpha_a_err, accel_a_err, vel_a_err = testhelp.generateError(
        alpha_a, 0, 0, accel_a, 0.0, 0.5, vel_a, 0.01, 0.01)

    plt.figure()
    plt.subplot(311)
    plt.plot(alpha_a)
    plt.plot(alpha_a_err)
    plt.subplot(312)
    plt.plot(accel_a)
    plt.plot(accel_a_err)
    plt.subplot(313)
    plt.plot(vel_a)
    plt.plot(vel_a_err)
    # plt.show()s

    return alpha_a, alpha_a_err, accel_a, accel_a_err, vel_a, vel_a_err


def getStateCovariance():
    P = np.matrix([[10.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [
                  0, 0, 1.0, 0, 0], [0, 0, 0, 0.0, 0], [0, 0, 0, 0, math.pi/45]])
    # math.pi/180
    return P


def main():
    print("Start systemmodel main")
    timestep = 0.5
    car1 = Vechile(26, timestep, x=0, y=0, gamma=0.0)
    car1Err = Vechile(26, timestep, x=0, y=0, gamma=0.0)

    l_input = Vechile.Input(2.0, 2.0)

    alpha_a, alpha_a_err, accel_a, accel_a_err, vel_a, vel_a_err = testSecvenceGenerate(
        timestep)

    X_car1 = []
    Y_car1 = []
    Gamma_car1 = []

    X_car1Err = []
    Y_car1Err = []
    Gamma_car1Err = []
    P = getStateCovariance()

    #
    plt.figure()
    ax = plt.subplot(111, aspect='equal')
    #
    e1 = plotHelp.plotEllipse(0, 0, P[1:3, 1:3])
    ax.add_artist(e1)
    X_car1Err.append(0)
    Y_car1Err.append(0)

    for accel, alpha, accel_err, alpha_err in zip(accel_a, alpha_a, accel_a_err, alpha_a_err):
        l_input = Vechile.Input(accel, alpha)
        newState = car1.f(car1.state, l_input)
        car1.state = newState
        X_car1.append(newState.x)
        Y_car1.append(newState.y)
        Gamma_car1.append(newState.gamma)

        newStateErr = car1Err.f(car1Err.state, l_input)
        car1Err.state = newStateErr

        F = car1Err.F(car1Err.state, l_input)
        P = F*P*np.transpose(F)
        e1 = plotHelp.plotEllipse(newStateErr.x, newStateErr.y, P[1:3, 1:3])
        ax.add_artist(e1)
        X_car1Err.append(newState.x)
        Y_car1Err.append(newState.y)
        Gamma_car1Err.append(newState.gamma)
    #
    # # ax.set_ylim(-140, 140)
    # # ax.set_xlim(-40, 240)
    #
    ax.set_ylim(-120, 120)
    ax.set_xlim(-10, 230)
    # print(X_car1, Y_car1)
    plt.plot(X_car1, Y_car1, '--ro')
    plt.plot(X_car1Err, Y_car1Err, '--go')
    plt.figure()
    plt.subplot(111)
    plt.plot(Gamma_car1)
    plt.plot(Gamma_car1Err)
    #
    plt.show()

    print("End systemmodel main")


if __name__ == '__main__':
    main()
