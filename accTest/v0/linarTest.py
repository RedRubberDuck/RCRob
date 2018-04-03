import numpy as np
import math
from matplotlib import pyplot as plt
import testhelp
import plotHelp
from SystemModel import *
from KalmanFilter import *


def main():
    timestep = 0.1
    car1 = Vechile(26, timestep, x=0, y=0, gamma=math.pi/4)
    car2 = Vechile(26, timestep, x=0, y=0, gamma=math.pi/4)
    alpha_a, accel_a = testhelp.generateInputSemnal(timestep)

    l_oldInput = Vechile.Input(accel_a[0], alpha_a[0])
    newState = car1.f(car1.state, l_oldInput)
    prevState = car1.state.X
    car1.state.X = newState.X
    car2.state.X = newState.X

    print('SSSS', car1.state.X, car2.state.X)

    alpha_a = alpha_a[1:]
    accel_a = accel_a[1:]

    V = []
    X = []
    Y = []
    W = []
    G = []
    Vl = []
    Xl = []
    Yl = []
    Wl = []
    Gl = []
    s = 0
    for alpha, accel in zip(alpha_a, accel_a):
        # print('AA:', alpha, accel)
        # break
        # sssssssssssssssss
        print(car2.state.w, car1.state.w)

        l_newInput = Vechile.Input(accel_f=accel, alpha=alpha)
        newCar1State = car1.f(car1.state, l_newInput)
        car1.state = newCar1State

        V.append(newCar1State.vf)
        X.append(newCar1State.x)
        Y.append(newCar1State.y)
        W.append(newCar1State.w)
        G.append(newCar1State.gamma)

        DX = car2.state.X - prevState
        DU = l_newInput._U - l_oldInput._U

        F_x = car2.F_x(car2.state, l_oldInput)

        F_u = car2.F_u(car2.state, l_oldInput)

        newCar2StateX = car2.state.X+F_x*DX+F_u*DU
        prevState = car2.state.X
        l_oldInput = l_newInput
        car2.state.X = newCar2StateX

        Vl.append(car2.state.vf)
        Xl.append(car2.state.x)
        Yl.append(car2.state.y)
        Wl.append(car2.state.w)
        Gl.append(car2.state.gamma)

        print(F_x, F_u, DX, DU)
        s += 1
        if s > 10:
            break

    plt.figure()

    plt.subplot(511)
    plt.plot(V)
    plt.plot(Vl)
    # plt.legend(['Init', 'Linear'])

    plt.subplot(512)
    plt.plot(X)
    plt.plot(Xl)
    # plt.legend(['Init', 'Linear'])
    plt.subplot(513)
    plt.plot(Y)
    plt.plot(Yl)
    # plt.legend(['Init', 'Linear'])
    plt.subplot(514)
    plt.plot(W)
    plt.plot(Wl)
    # plt.legend(['Init', 'Linear'])
    plt.subplot(515)
    plt.plot(G)
    plt.plot(Gl)
    plt.legend(['Init', 'Linear'])
    plt.show()

    # print(Xl, Yl)


if __name__ == "__main__":
    main()
