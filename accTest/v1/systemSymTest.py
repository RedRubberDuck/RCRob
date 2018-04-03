import numpy as np
import math
from systemSym import Vehicle


from sympy import symbols, Matrix
import sympy
from matplotlib import pyplot as plt


def test(dt, wheelbase):
    alpha, a, x, y, v, w, theta, dtheta, time = symbols('alpha, a, x, y, v, w, theta, dtheta, t')

    subs = {x: 0, y: 0, v: 0, a: 0, alpha: 0, time: dt, w: wheelbase, theta: 0}

    d = v*time
    beta = (d/w)*sympy.tan(alpha)
    r = w/sympy.tan(alpha)
    fxu = Matrix(
        [[v+a*time],
         [x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
         [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
         [theta + beta]])

    F_x = fxu.jacobian(Matrix([v, x, y, theta]))
    F_u = fxu.jacobian(Matrix([a, alpha]))
    return x, y, v, a, alpha, theta, subs, F_x, F_u


def main():
    x_s, y_s, v_s, a_s, alpha_s, theta_s, subs_s, F_x_s, F_u_s = test(0.1, 26)
    car1 = Vehicle(wheelbase=26, timestep=0.1)
    car2 = Vehicle(wheelbase=26, timestep=0.1)
    car3 = Vehicle(wheelbase=26, timestep=0.1)

    acc = 200
    alpha = math.pi/180*10
    l_input = np.matrix([[acc], [alpha]])
    newX = car1.f(car1.X, l_input)

    prevX = car1.X
    prevX2 = car1.X
    prevU = l_input

    car1.X = newX
    car2.X = newX
    car3.X = newX

    Vf = []
    Px = []
    Py = []
    T = []

    Vf_l = []
    Px_l = []
    Py_l = []
    T_l = []

    Vf_l2 = []
    Px_l2 = []
    Py_l2 = []
    T_l2 = []

    for i in range(100):
        l_input = np.matrix([[0.0], [alpha]])
        # Simulate--------------------------------------------------------------
        newX = car1.f(car1.X, l_input)
        car1.X = newX

        Vf.append(newX[0, 0])
        Px.append(newX[1, 0])
        Py.append(newX[2, 0])
        T.append(newX[3, 0])

        # Linarity test---------------------------------------------------------
        dX = car2.X - prevX
        dU = l_input - prevU
        # ----------------------------------------------------------------------
        Fx = car2.Fx(car2.X, prevU)
        Fu = car2.Fu(car2.X, prevU)
        # ----------------------------------------------------------------------
        newX = car2.X + Fx*dX + Fu*dU
        prevX = car2.X
        prevU = l_input

        car2.X = newX

        # --------------------------------------------------------------------W--
        Vf_l.append(newX[0, 0])
        Px_l.append(newX[1, 0])
        Py_l.append(newX[2, 0])
        T_l.append(newX[3, 0])

        # ----------------------------------------------------------------------

        subs_s[v_s] = car3.X[0, 0]
        subs_s[x_s] = car3.X[1, 0]
        subs_s[y_s] = car3.X[2, 0]
        subs_s[theta_s] = car3.X[3, 0]
        subs_s[a_s] = prevU[0, 0]
        subs_s[alpha_s] = prevU[1, 0]

        Fx_s = np.array(F_x_s.evalf(subs=subs_s)).astype(float)
        Fu_s = np.array(F_u_s.evalf(subs=subs_s)).astype(float)

        newX = car3.X+Fx_s*dX+Fu_s*dU
        prevX2 = car3.X
        car3.X = newX

        Vf_l2.append(newX[0, 0])
        Px_l2.append(newX[1, 0])
        Py_l2.append(newX[2, 0])
        T_l2.append(newX[3, 0])

    alpha = -math.pi/180*3
    for i in range(100):
        l_input = np.matrix([[0.0], [alpha]])
        # Simulate--------------------------------------------------------------
        newX = car1.f(car1.X, l_input)
        # print((newX-car1.X)[3, 0])
        car1.X = newX

        Vf.append(newX[0, 0])
        Px.append(newX[1, 0])
        Py.append(newX[2, 0])
        T.append(newX[3, 0])

        # Linarity test---------------------------------------------------------
        dX = car2.X - prevX
        dU = l_input - prevU

        # ----------------------------------------------------------------------
        Fx = car2.Fx(car2.X, prevU)
        Fu = car2.Fu(car2.X, prevU)
        # ----------------------------------------------------------------------

        newX = car2.X + Fx*dX + Fu*dU
        # print((Fx*dX + Fu*dU)[3, 0])
        prevX = car2.X
        prevU = l_input

        car2.X = newX
        # ----------------------------------------------------------------------
        Vf_l.append(newX[0, 0])
        Px_l.append(newX[1, 0])
        Py_l.append(newX[2, 0])
        T_l.append(newX[3, 0])

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        subs_s[v_s] = car3.X[0, 0]
        subs_s[x_s] = car3.X[1, 0]
        subs_s[y_s] = car3.X[2, 0]
        subs_s[theta_s] = car3.X[3, 0]
        subs_s[a_s] = prevU[0, 0]
        subs_s[alpha_s] = prevU[1, 0]

        Fx_s = np.array(F_x_s.evalf(subs=subs_s)).astype(float)
        Fu_s = np.array(F_u_s.evalf(subs=subs_s)).astype(float)

        newX = car3.X+Fx_s*dX+Fu_s*dU
        prevX2 = car3.X
        car3.X = newX

        Vf_l2.append(newX[0, 0])
        Px_l2.append(newX[1, 0])
        Py_l2.append(newX[2, 0])
        T_l2.append(newX[3, 0])

    plt.figure()
    # plt.subplot(211)
    # plt.plot(Vf)
    plt.plot(T, '--b')
    # plt.plot(Px, Py, '--b')
    # plt.subplot(212)
    # plt.plot(Vf_l)
    plt.plot(T_l)
    plt.plot(T_l2)
    # plt.plot(Px_l, Py_l, '--r')
    # plt.figure()

    plt.show()


if __name__ == "__main__":
    main()
