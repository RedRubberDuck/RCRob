import sympy
import numpy as np

from matplotlib import pyplot as plt


Aax, Aay, Vax, Vay, x, y, Vff, theta, dtheta, alpha, Vfb, dt, w = sympy.symbols(
    'Aax, Aay, Vax, Vay, x, y, Vff, theta, dtheta, alpha, Vfb, dt, w')


def systemModelDef():
    print("Definition")
    # Aax, Aay, Vax, Vay, x, y, Vff, theta, dtheta, alpha, Vfb, dt, w = sympy.symbols(
    #     'Aax, Aay, Vax, Vay, x, y, Vff, theta, dtheta, alpha, Vfb, dt, w')

    print('Symbols:', Aax, Aay, Vax, Vay, x, y, Vff, theta, dtheta, alpha, Vfb, dt, w)
    f_xu = sympy.Matrix([[Vax + dt * Aax],
                         [Vay + dt * Aay],
                         [sympy.sqrt((Vax + dt * Aax)**2+(Vay + dt * Aay)**2)
                          * Vay/sympy.Abs(Vay)*sympy.cos(alpha)],
                         #  [sympy.atan((Vax + dt * Aax)/(Vay + dt * Aay))],
                         [Vfb/w*sympy.tan(alpha)],
                         [theta + dt * dtheta],
                         [x + Vfb * dt * sympy.cos(theta)],
                         [y + Vfb * dt * sympy.sin(theta)]
                         ])

    f_xu1 = sympy.Matrix([[Vax + dt * Aax],
                          [Vay + dt * Aay],
                          [sympy.sqrt((Vax + dt * Aax)**2+(Vay + dt * Aay)**2)
                           * sympy.cos(alpha)],
                          #  [sympy.atan((Vax + dt * Aax)/(Vay + dt * Aay))],
                          [Vfb/w*sympy.tan(alpha)],
                          [theta + dt * dtheta],
                          [x + Vfb * dt * sympy.cos(theta)],
                          [y + Vfb * dt * sympy.sin(theta)]
                          ])

    print("f_xu.:", f_xu)
    F_x = f_xu1.jacobian(sympy.Matrix([[Vax], [Vay], [Vfb],  [dtheta], [theta], [x], [y]]))
    print("F_x", F_x)
    F_u = f_xu1.jacobian(sympy.Matrix([[Aax], [Aay], [alpha]]))
    print("F_u", F_u)
    subs = {Aax: 0.0, Aay: 0.0, Vax: 0.0, Vay: 0.02, x: 0.0, y: 0.0, Vff: 0.0,
            theta: np.radians(45), dtheta: 0.0, alpha: 0.0, Vfb: 0.0, dt: 0.02, w: 0.26}

    return f_xu, F_x, F_u, subs


def systemSim(f_xu, subs):
    print("Simulate system")

    Acc = -10.0
    steer = np.radians(-23)

    Acx = np.sin(steer)*Acc
    Acy = np.cos(steer)*Acc
    print('AAAA', Acx, Acy)
    print('steer', steer)

    index = 0

    subs[Aax] = Acx
    subs[Aay] = Acy
    # subs[alpha] = np.radians(23)

    Acc_c = None
    X_c = None
    for i in range(247):
        X = f_xu.evalf(subs=subs)
        subs[alpha] = steer

        subs[Vax] = X[0, 0]
        subs[Vay] = X[1, 0]
        subs[Vfb] = X[2, 0]
        # subs[alpha] = X[3, 0]
        subs[dtheta] = X[3, 0]
        subs[theta] = X[4, 0]
        subs[x] = X[5, 0]
        subs[y] = X[6, 0]

        if (index == 0):
            X_c = np.matrix(X)
            Acc_c = np.matrix([[Acx], [Acy], [steer]])
            subs[Aax] = 0.0
            subs[Aay] = 0.0

        else:
            X_c = np.concatenate((X_c, np.matrix(X)), axis=1)
            Acc_c = np.concatenate((Acc_c, np.matrix([[0.0], [0.0], [steer]])), axis=1)
        index += 1

    # print('X_c:', X_c)
    # plt.figure()
    # plt.plot(X_c[6, :].tolist()[0], X_c[7, :].tolist()[0])
    # plt.show()
    # print(X_c[1, :].tolist())
    # print(X_c[3, :].tolist()[0], X_c[4, :].tolist()[0])
    return X_c, Acc_c


def systemLinearSim(F_x, F_u, Acc, subs):
    prevX = curX = np.zeros((7, 1))
    prevX[1, 0] = curX[1, 0] = 0.01
    prevX[4, 0] = curX[4, 0] = np.radians(45)
    # curX[1, 0] = prevX[1, 0] = 0.02
    prevU = np.zeros((3, 1))
    X_linSim = prevX
    index = 0
    for acc in Acc.T:
        dX = curX - prevX
        dU = acc.T - prevU

        subs[Vax] = curX[0, 0]
        # if(curX[1, 0] == 0.0):
        #     subs[Vay] = 0.02
        # else:
        subs[Vay] = curX[1, 0]
        subs[Vfb] = curX[2, 0]
        # subs[alpha] = curX[3, 0]
        subs[dtheta] = curX[3, 0]
        subs[theta] = curX[4, 0]
        subs[x] = curX[5, 0]
        subs[y] = curX[6, 0]

        subs[Aax] = prevU[0, 0]
        subs[Aay] = prevU[1, 0]
        subs[alpha] = prevU[2, 0]

        F_x_v = np.matrix(F_x.evalf(subs=subs))
        F_u_v = np.matrix(F_u.evalf(subs=subs))

        newX = curX + F_x_v * dX + F_u_v * dU
        # print('DX', F_x_v*dX)
        # print('DU', F_u_v*dU)

        prevX = curX
        curX = newX
        prevU = acc.T

        X_linSim = np.concatenate((X_linSim, newX), axis=1)
        index += 1
        if index < 4:
            print('AAA', F_x_v * dX + F_u_v * dU)
        # print('AAA', prevX)
    print(X_linSim)
    return X_linSim


def plotting(X_sim, X_linSim):

    plt.figure()
    plt.plot(X_sim[0, :].tolist()[0])
    plt.plot(X_linSim[0, :].tolist()[0])
    plt.legend(['Sim', 'Lin'])
    plt.figure()
    plt.plot(X_sim[1, :].tolist()[0])
    plt.plot(X_linSim[1, :].tolist()[0])
    plt.legend(['Sim', 'Lin'])

    plt.figure()
    plt.plot(X_sim[5, :].tolist()[0], X_sim[6, :].tolist()[0])
    plt.plot(X_linSim[5, :].tolist()[0], X_linSim[6, :].tolist()[0])
    plt.legend(['Sim', 'Lin'])
    # plt.plot(X_linSim[3, :].tolist()[0])
    plt.show()


def main():
    print("System Model")
    f_xu, F_x, F_u, subs = systemModelDef()
    X_sim, Acc_c = systemSim(f_xu, subs)
    X_linSim = systemLinearSim(F_x, F_u, Acc_c, subs)
    # X_linSim = None
    plotting(X_sim, X_linSim)


if __name__ == '__main__':
    main()
