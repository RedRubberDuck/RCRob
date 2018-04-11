import time
import numpy as np

from matplotlib import pyplot as plt


class RobotSystem:

    def __init__(self, wheelbase, dt):
        self.__dt = dt
        self.__wb = wheelbase

    def move(self, U, X):
        X = X + np.matrix([
            [0.0],
            [0.0],
            [X[0, 0]*self.__dt*np.cos(X[5, 0])],
            [X[0, 0]*self.__dt*np.sin(X[5, 0])],
            [0.0],
            [X[4, 0]*self.__dt]
        ])
        X[4, 0] = X[0, 0]/self.__wb*np.tan(X[1, 0])
        X[0, 0] = U[0, 0]
        X[1, 0] = U[1, 0]
        return X

    def F_x(self, U, X):
        return np.matrix([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [self.__dt*np.cos(X[5, 0]), 0.0, 1.0, 0.0, 0.0, -self.__dt*X[0, 0]*np.sin(X[5, 0])],
            [self.__dt*np.sin(X[5, 0]), 0.0, 0.0, 1.0, 0.0, self.__dt*X[0, 0]*np.cos(X[5, 0])],
            [np.tan(X[1, 0])/self.__wb, X[0, 0]/self.__wb *
             (np.tan(X[1, 0])**2+1), 0.0, 0.0, 0.0, 0.0],
            #  X[0, 0]/self.__wb * (np.tan(X[1, 0])**2+1)
            [0.0, 0.0, 0.0, 0.0, self.__dt, 1.0]
        ])

    def F_u(self, U, X):
        return np.matrix([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ])

    def h(self, X):
        return np.matrix([[X[0, 0]], [X[4, 0]]])

    def H_x(self, X):
        return np.matrix([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])


def simulate():
    rob1 = RobotSystem(0.265, 0.05)
    u = np.matrix([[0.2], [np.radians(0)]])

    X = np.zeros((6, 1))
    X_a = X.copy()
    X_a_l = X.copy()
    Y_a = rob1.h(X)

    PrevX = CurX = np.zeros((6, 1))
    PrevU = np.zeros((2, 1))

    for i in range(450):
        newX = rob1.move(u, X)
        X_a = np.concatenate((X_a, newX), axis=1)
        X = newX
        Y = rob1.h(X)
        Y_a = np.concatenate((Y_a, Y), axis=1)

        f_x = rob1.F_x(u, PrevX)
        f_u = rob1.F_u(u, PrevX)

        dX = CurX - PrevX
        dU = u - PrevU
        if dX[1, 0] != 0.0:
            print('AAA', dX, 'BBB', f_x * dX, f_u * dU)
        # print(f_x * dX, f_u * dU)
        newX = CurX + f_x * dX + f_u * dU
        PrevX = CurX
        CurX = newX
        PrevU = u.copy()
        X_a_l = np.concatenate((X_a_l, newX), axis=1)

        if i < 23:
            # u[0, 0] = -0.2
            u[1, 0] = u[1, 0] + np.radians(2.0)
    # print(new)

    # print(Y_a)
    plt.figure()
    plt.plot(X_a[2, :].tolist()[0], X_a[3, :].tolist()[0])
    plt.plot(X_a_l[2, :].tolist()[0], X_a_l[3, :].tolist()[0])

    plt.figure()
    plt.plot(X_a[0, :].tolist()[0])
    plt.plot(Y_a[0, :].tolist()[0])
    plt.plot(X_a_l[0, :].tolist()[0])

    plt.figure()
    plt.plot(X_a[1, :].tolist()[0])
    plt.plot(X_a_l[1, :].tolist()[0])
    plt.legend(['Sim', 'Lin'])

    plt.figure()
    plt.plot(X_a[4, :].tolist()[0])
    plt.plot(X_a_l[4, :].tolist()[0])
    plt.legend(['Sim', 'Lin'])

    plt.figure()
    plt.plot(X_a[5, :].tolist()[0])
    plt.plot(X_a_l[5, :].tolist()[0])
    plt.legend(['Sim', 'Lin'])
    plt.show()


import sympy


def main():
    print('System model')
    simulate()
    vf, wb, alpha = sympy.symbols(' vf, wb, alpha')

    dtheta = sympy.Matrix([vf/wb*sympy.tan(alpha)])

    f_d = dtheta.jacobian(sympy.Matrix([vf, alpha]))
    print('F_d', f_d)


if __name__ == '__main__':
    main()
