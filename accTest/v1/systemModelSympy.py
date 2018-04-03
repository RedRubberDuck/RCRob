import sympy
import numpy as np

from matplotlib import pyplot as plt


def dTheteDiff():
    v, x, y, dtheta, theta, alpha, a, dt, w, temp = sympy.symbols(
        'v, x, y, dtheta, theta, alpha, a, dt, w,temp')
    r = w/sympy.tan(alpha)
    f = sympy.Matrix([[x - r*sympy.sin(theta) + r * sympy.sin(theta+dtheta*dt)],
                      [y + r*sympy.cos(theta) - r * sympy.cos(theta+dtheta*dt)],
                      [v + a * dt],
                      [theta+dtheta*dt],
                      [dtheta]])
    Fx = f.jacobian(sympy.Matrix([temp, temp, temp, temp, dtheta]))
    return Fx


def main():

    v, x, y, dtheta, theta, alpha, a, dt, w, temp = sympy.symbols(
        'v, x, y, dtheta, theta, alpha, a, dt, w,temp')

    # d = v
    r = w/sympy.tan(alpha)
    # dtheta = (v+a/2*dt)/w*sympy.tan(alpha)
    # dtheta = v/w*sympy.tan(alpha)

    f = sympy.Matrix([[x - r*sympy.sin(theta) + r * sympy.sin(theta+dtheta)],
                      [y + r*sympy.cos(theta) - r * sympy.cos(theta+dtheta)],
                      [v + a * dt],
                      [theta+dtheta],
                      [(v*dt+a/2*dt**2)/w*sympy.tan(alpha)]])
    print('f', f)

    Fx = f.jacobian(sympy.Matrix([x, y, v, theta, dtheta]))
    Fu = f.jacobian(sympy.Matrix([a, alpha]))

    print('---------------------------------------------------------------------')
    print('Fx', Fx)
    print('---------------------------------------------------------------------')

    subs = {x: 0, y: 0, v: 0, theta: 0, dtheta: 0, alpha: np.radians(20), a: 100, dt: 0.01, w: 26}
    prevX = np.matrix([[subs[x]], [subs[y]], [0], [subs[theta]], [subs[dtheta]]])
    X = np.matrix(f.evalf(subs=subs))
    X_a = X
    X_al = X

    prevU = np.matrix([[subs[a]], [subs[alpha]]])
    subs[a] = 0.0
    subs[x] = X[0, 0]
    subs[y] = X[1, 0]
    subs[v] = X[2, 0]
    subs[theta] = X[3, 0]
    subs[dtheta] = X[4, 0]

    for i in range(150):
        newX = np.matrix(f.evalf(subs=subs))
        X_a = np.concatenate((X_a, newX), axis=1)

        Fx_v = np.matrix(Fx.evalf(subs=subs))
        Fu_v = np.matrix(Fu.evalf(subs=subs))
        u = np.matrix([[subs[a]], [subs[alpha]]])

        subs[x] = newX[0, 0]
        subs[y] = newX[1, 0]
        subs[v] = newX[2, 0]
        subs[theta] = newX[3, 0]
        subs[dtheta] = newX[4, 0]

        dU = u - prevU
        dX = X - prevX
        newX2 = X + Fx_v*dX + Fu_v*dU

        X_al = np.concatenate((X_al, newX2), axis=1)
        prevX = X
        prevU = u
        X = newX2
        print('X', abs(newX[0, 0] - newX2[0, 0]))
        print('Y', abs(newX[1, 0] - newX2[1, 0]))

    plt.figure()
    plt.plot(X_a[2, :].tolist()[0], '--g')
    plt.plot(X_al[2, :].tolist()[0], '--r')

    plt.figure()
    plt.plot(X_a[3, :].tolist()[0], '--g')
    plt.plot(X_al[3, :].tolist()[0], '--r')

    plt.figure()
    plt.plot(X_a[4, :].tolist()[0])
    plt.plot(X_al[4, :].tolist()[0])
    plt.figure()
    # plt.subplot(111,eq)
    plt.plot(X_a[0, :].tolist()[0], X_a[1, :].tolist()[0], '--b')
    plt.plot(X_al[0, :].tolist()[0], X_al[1, :].tolist()[0], '--g')
    plt.show()


if __name__ == "__main__":
    main()
