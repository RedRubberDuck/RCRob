import numpy as np
import math
from matplotlib import pyplot as plt
import testhelp
import plotHelp
from SystemModel import *
from KalmanFilter import *


def testSecvenceGenerate(timestep):
    alpha_a, accel_a = testhelp.generateInputSemnal(timestep)
    alpha_a_err, accel_a_err = testhelp.generateInputError(
        alpha_a, 0.0, 6.5, accel_a, 0.00, 0.4)

    plt.figure()
    plt.subplot(211)
    plt.plot(alpha_a)
    plt.plot(alpha_a_err)
    plt.subplot(212)
    plt.plot(accel_a)
    plt.plot(accel_a_err)

    return alpha_a, alpha_a_err, accel_a, accel_a_err


def main():
    print("Start systemmodel main")

    timestep = 0.1
    car1 = Vechile(26, timestep, x=0, y=0, gamma=math.pi/4)
    car1Err = Vechile(26, timestep, x=0, y=0, gamma=math.pi/4)

    alpha_a, alpha_a_err, accel_a, accel_a_err = testSecvenceGenerate(
        timestep)

    # The vechile position and oriantation without error
    X_car1 = []
    Y_car1 = []
    Gamma_car1 = []
    W_car1 = [0]

    # The vechile position and oriantation with error
    X_car1Err = []
    Y_car1Err = []
    Gamma_car1Err = []
    W_car1Err = [0]

    # The vechile position and oriantation with error
    X_car1ErrF = []
    Y_car1ErrF = []
    Gamma_car1ErrF = []
    W_car1ErrF = [0]

    plt.figure()
    ax = plt.subplot(111, aspect='equal')

    X_car1ErrF.append(0)
    Y_car1ErrF.append(0)
    Gamma_car1ErrF.append(math.pi/4)

    X_car1Err.append(0)
    Y_car1Err.append(0)
    Gamma_car1Err.append(math.pi/4)

    X_car1.append(0)
    Y_car1.append(0)
    Gamma_car1.append(math.pi/4)

    # State covariance
    P = getStateCovariance()
    Q = getProcessNoise(0.1, timestep, 6.5, 26, 1.0)
    R = getMeasurmentNoise(velf=0.1, groErr=0.05)
    # e1 = plotHelp.plotEllipse(0, 0, P[1:3, 1:3])
    # ax.add_artist(e1)

    l_kalmanFilter = KalmanFilter(car1Err.state, car1Err.f, car1Err.h, car1Err.F_x, car1Err.F_u,
                                  car1Err.H, P, Q, R, Vechile.State, Vechile.Output)

    Velf = [0]
    VelfErr = [0]
    VelfErrF = [0]

    for accel, alpha in zip(accel_a, alpha_a):
        l_input = Vechile.Input(accel, alpha)

        newState = car1.f(car1.state, l_input)
        l_measurment = car1.h(car1.state)
        car1.state = newState

        # Position without error
        X_car1.append(newState.x)
        Y_car1.append(newState.y)
        Gamma_car1.append(newState.gamma)
        Velf.append(newState.vf)
        W_car1.append(newState.w)

    Vf_err, W_err = testhelp.generateMesError(Velf, 0.0, 0.5, W_car1, 0.00, 0.05)

    for accel_err, alpha_err, vf_mes, w_mes in zip(accel_a_err, alpha_a_err, Vf_err, W_err):
        l_measurment = Vechile.Output(vf_mes, w_mes)
        l_input = Vechile.Input(accel_err, alpha_err)
        newStateErr = car1Err.f(car1Err.state, l_input)
        car1Err.state = newStateErr

        X_car1Err.append(newStateErr.x)
        Y_car1Err.append(newStateErr.y)
        Gamma_car1Err.append(newStateErr.gamma)
        VelfErr.append(newStateErr.vf)
        W_car1Err.append(newStateErr.w)

        newStateErrF, l_P = l_kalmanFilter.predict(l_kalmanFilter.X, l_input, l_kalmanFilter.P)

        newStateErrF, l_P = l_kalmanFilter.update(newStateErrF, l_measurment, l_P)
        l_kalmanFilter.X = newStateErrF
        l_kalmanFilter.P = l_P

        X_car1ErrF.append(newStateErrF.x)
        Y_car1ErrF.append(newStateErrF.y)
        Gamma_car1ErrF.append(newStateErrF.gamma)
        VelfErrF.append(newStateErrF.vf)
        W_car1ErrF.append(newStateErrF.w)

    plt.plot(X_car1, Y_car1, '--ro')
    plt.plot(X_car1Err, Y_car1Err, '--go')
    plt.plot(X_car1ErrF, Y_car1ErrF, '--bo')
    plt.legend(['Real', 'Error', 'Filtered'])

    plt.figure()
    plt.title('Angle speed')
    plt.plot(W_car1)
    plt.plot(W_err)
    plt.plot(W_car1Err)
    plt.plot(W_car1ErrF)
    plt.legend(['Real', 'Added Error', 'Error based pred.', 'Filtered'])

    plt.figure()
    plt.title('Oriantation')
    plt.plot(Gamma_car1)
    plt.plot(Gamma_car1Err)
    plt.plot(Gamma_car1ErrF)
    plt.legend(['Real', 'Error', 'Filtered'])

    plt.figure()
    plt.title('Velocity')
    plt.plot(Velf)
    plt.plot(Vf_err)
    plt.plot(VelfErr)
    plt.plot(VelfErrF)
    plt.legend(['Real', 'Added Error', 'Error based pred.', 'Filtered'])

    posErr = []
    posErrF = []

    for x_p, y_p, x_pf, y_pf, x_pR, y_pR in zip(X_car1, Y_car1, X_car1ErrF, Y_car1ErrF, X_car1Err, Y_car1Err):
        p = complex(x_p, y_p)
        pf = complex(x_pf, y_pf)
        pr = complex(x_pR, y_pR)
        errorF = abs(p-pf)
        error = abs(p-pr)

        posErr.append(error)
        posErrF.append(errorF)

        # #

    plt.figure()
    plt.plot(posErr)
    plt.plot(posErrF)
    plt.legend(['Error', 'ErrorF'])
    plt.show()

    print("End systemmodel main")


if __name__ == '__main__':
    main()
