
import numpy as np
import math
from matplotlib import pyplot as plt
import testhelp
import plotHelp
from SystemModel import *
from KalmanFilter import *


def main():
    timestep = 0.05
    car1 = Vechile(26, timestep, x=0, y=0, gamma=math.pi/4)
    car2 = Vechile(26, timestep, x=0, y=0, gamma=math.pi/4)

    alpha_a, accel_a = testhelp.generateInputSemnal(timestep)

    X = []
    Y = []
    W = []
    XL = []
    YL = []
    WL = []
    GammaL = []

    index = 0

    for alpha, accel in zip(alpha_a, accel_a):
        l_input = Vechile.Input(alpha=alpha, accel_f=accel)
        newState = car1.f(car1.state, l_input)
        car1.state = newState

        X.append(car1.state.x)
        Y.append(car1.state.y)
        W.append(car1.state.w)
        F_x = car2.F_x(car2.state, l_input)
        F_u = car2.F_u(car2.state, l_input)

        print('State:', F_x*car2.state.X)
        print('State:', F_u*l_input._U)

        newX = F_x*car2.state.X+F_u*l_input._U

        car2.state.X = newX
        XL.append(car2.state.x)
        YL.append(car2.state.y)
        WL.append(car2.state.w)

        # index += 1
        # if index > 10:
        #     break
    # print(GammaL)
    plt.plot(X, Y)
    plt.plot(XL, YL)
    plt.figure()
    plt.plot(W)
    plt.plot(WL)
    # plt.plot(GammaL)
    plt.show()


if __name__ == "__main__":
    main()
