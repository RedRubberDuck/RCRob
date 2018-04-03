# import
from filterpy.kalman import ExtendedKalmanFilter as EKF
from numpy import dot, array, sqrt, matrix, cos, sin, tan
import sympy
import math
import numpy as np
from sympy import symbols, Matrix
from matplotlib import pyplot as plt

import plotHelp


class RobotEKF2(EKF):
    def __init__(self, dt, wheelbase, std_acc, std_steer):
        EKF.__init__(self, 5, 1, 2)
        self.dt = dt
        self.wheelbase = wheelbase
        self.std_acc = std_acc
        self.std_steer = std_steer

        self.M = array([[self.std_acc**2, 0], [0, self.std_steer**2]])

    def move(self, x, u, dt):
        hdg = x[3, 0]
        a = u[0,    0]
        vel = x[2, 0]
        dtheta = x[4, 0]
        steering_angle = u[1, 0]
        dist = vel
        x[4, 0] = 0.0
        if abs(steering_angle) > 0.01:
            beta = (dist/self.wheelbase) * tan(steering_angle)
            r = self.wheelbase / tan(steering_angle)

            dx = np.matrix([
                [-r*sin(hdg) + r*sin(hdg+dtheta*dt)],
                [r*cos(hdg) - r*cos(hdg+dtheta*dt)],
                [a*dt],
                [dtheta*dt],
                [beta]])
        else:
            dx = np.matrix([
                [vel*self.dt*cos(hdg)],
                [vel*self.dt*sin(hdg)],
                [a*dt],
                [dtheta],
                [0]])
        return x + dx

    def F_x(self, x, u):
        px = x[0, 0]
        py = x[1, 0]
        v = x[2, 0]
        theta = x[3, 0]
        dtheta = x[4, 0]
        alpha = u[1, 0]
        acc = u[0, 0]
        r = self.wheelbase/np.tan(alpha)

        if alpha < 0.0001:
            alpha = 0.0001

        return np.matrix([[1, 0, 0, -r*np.cos(theta)+r*np.cos(dtheta*self.dt+theta),
                           self.dt*r*np.cos(dtheta*self.dt+theta)],
                          [0, 1, 0, -r*np.sin(theta)+r*np.sin(dtheta*self.dt+theta),
                           self.dt*r*np.sin(dtheta*self.dt+theta)],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, self.dt],
                          [0, 0, np.tan(alpha)/self.wheelbase, 0, 0]])

    def F_u(self, x, u):
        px = x[0, 0]
        py = x[1, 0]
        v = x[2, 0]
        theta = x[3, 0]
        dtheta = x[4, 0]
        alpha = u[1, 0]
        acc = u[0, 0]
        if alpha < 0.0001:
            alpha = 0.0001

        return np.matrix([
            [0, -self.wheelbase*(-np.tan(alpha)**2-1)*np.sin(theta)/np.tan(
                alpha)**2 + self.wheelbase*(-np.tan(alpha)**2-1)*np.sin(theta+dtheta*self.dt)/np.tan(alpha)**2],
            [0, self.wheelbase*(-np.tan(alpha)**2-1)*np.cos(theta)/np.tan(
                alpha)**2 - self.wheelbase*(-np.tan(alpha)**2-1)*np.cos(theta+dtheta*self.dt)/np.tan(alpha)**2],
            [self.dt, 0],
            [0, 0],
            [0, v*(np.tan(alpha)**2+1)/self.wheelbase]])

    def predict(self, u=array([[0.0], [0.0]])):
        newX = self.move(self.x, u, self.dt)

        #
        F_x = self.F_x(self.x, u)
        F_u = self.F_u(self.x, u)

        self.P = dot(F_x, self.P).dot(F_x.T) + dot(F_u, self.M).dot(F_u.T)
        self.x = newX

    def Hx(X):
        return array([[X[2, 0]], [X[4, 0]]])

    def H_of(X):
        return array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])

    def update(self, z, R):
        super(RobotEKF2, self).update(z=z, HJacobian=RobotEKF2.H_of, Hx=RobotEKF2.Hx, R=R)


class RobotEKF(EKF):
    def __init__(self, dt, wheelbase, std_acc, std_steer):
        EKF.__init__(self, 5, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase
        self.std_acc = std_acc
        self.std_steer = std_steer

        alpha, a, x, y, v, w, theta, dtheta, time = symbols(
            'alpha, a, x, y, v, w, theta, dtheta, t')

        d = v
        # + a/2*time**2
        beta = (d/w)*sympy.tan(alpha)
        r = w/sympy.tan(alpha)

        self.fxu = Matrix(
            [[x - r * sympy.sin(theta) + r * sympy.sin(theta + dtheta*time)],
                [y + r * sympy.cos(theta) - r * sympy.cos(theta + dtheta*time)],
                [v + a * time],
                [theta + dtheta*time],
                [beta]])

        self.F_x = self.fxu.jacobian(Matrix([x, y, v, theta, dtheta]))
        self.F_u = self.fxu.jacobian(Matrix([a, alpha]))

        print('F_x', sympy.simplify(self.F_x))
        print('F_u', sympy.simplify(self.F_u))

        self.subs = {x: 0, y: 0, v: 0, alpha: 0, a: 0, time: dt, w: wheelbase, theta: 0}
        self.x_x, self.x_y = x, y

        self.v, self.alpha, self.theta, self.dtheta, self.a = v, alpha, theta, dtheta, a

    def move(self, x, u, dt):
        hdg = x[3, 0]
        a = u[0,    0]
        vel = x[2, 0]
        dtheta = x[4, 0]
        steering_angle = u[1, 0]
        dist = vel
        x[4, 0] = 0.0
        if abs(steering_angle) > 0.001:
            beta = (dist/self.wheelbase) * tan(steering_angle)
            r = self.wheelbase / tan(steering_angle)

            dx = np.matrix([
                [-r*sin(hdg) + r*sin(hdg+dtheta*dt)],
                [r*cos(hdg) - r*cos(hdg+dtheta*dt)],
                [a*dt],
                [dtheta*dt],
                [beta]])
        else:
            dx = np.matrix([
                [vel*self.dt*cos(hdg)],
                [vel*self.dt*sin(hdg)],
                [a*dt],
                [dtheta],
                [0]])
        return x + dx

    def predict(self, u=array([[0.0], [0.0]])):
        self.x = self.move(self.x, u, self.dt)
        # Update the state value
        self.subs[self.alpha] = u[1]
        self.subs[self.a] = u[0]

        #
        F_x = array(self.F_x.evalf(subs=self.subs)).astype(float)
        F_u = array(self.F_u.evalf(subs=self.subs)).astype(float)

        self.subs[self.x_x] = self.x[0, 0]
        self.subs[self.x_y] = self.x[1, 0]
        self.subs[self.v] = self.x[2, 0]
        self.subs[self.theta] = self.x[3, 0]
        self.subs[self.dtheta] = self.x[4, 0]

        M = array([[self.std_acc**2, 0], [0, self.std_steer**2]])
        self.P = dot(F_x, self.P).dot(F_x.T) + dot(F_u, M).dot(F_u.T)

    def Hx(X):
        return array([[X[2, 0]], [X[4, 0]]])

    def H_of(X):
        return array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])

    def update(self, z, R):
        super(RobotEKF, self).update(z=z, HJacobian=RobotEKF.H_of, Hx=RobotEKF.Hx, R=R)


def generateTest(std_vel, std_accel, std_steer, std_theta, std_dtheta):
    rob1 = RobotEKF(dt=0.1, wheelbase=26, std_acc=std_accel, std_steer=std_steer)
    rob2 = RobotEKF2(dt=0.1, wheelbase=26, std_acc=std_accel, std_steer=std_steer)
    X = rob1.x
    X = rob2.x
    # State = X
    u = matrix([[90], [math.pi/180*20]])
    U_a = u

    X = rob1.move(x=X, u=u, dt=rob1.dt)
    # print('State', X)
    State = X
    # State = np.concatenate((State, X), axis=1)
    u = matrix([[0], [math.pi/180*2]])
    for i in range(100):

        X = rob1.move(x=X, u=u, dt=rob1.dt)
        U_a = np.concatenate((U_a, u), axis=1)
        State = np.concatenate((State, X), axis=1)

        # State.append(X)
    vel_err = State[2, :]+np.random.normal(0.0, std_vel, State.shape[1])
    theta_err = State[3, :]+np.random.normal(0.0, std_theta, State.shape[1])
    dtheta_err = State[4, :]+np.random.normal(0.0, std_dtheta, State.shape[1])

    plt.figure()
    plt.plot(theta_err[0, :].tolist()[0])
    plt.plot(State[3, :].tolist()[0])

    plt.figure()
    plt.plot(dtheta_err[0, :].tolist()[0])
    plt.plot(State[4, :].tolist()[0])
    plt.show()

    U_a[0, :] += np.random.normal(0.0, std_accel, U_a.shape[1])
    U_a[1, :] += np.random.normal(0.0, std_steer, U_a.shape[1])
    return State, U_a, rob1, rob2, vel_err, theta_err, dtheta_err


def main():
    std_vel = 0.0
    std_theta = np.radians(1)
    std_dtheta = np.radians(0.001)
    print("S", std_dtheta)
    State, U_a, rob1, rob2, vel_err, theta_err, dtheta_err = generateTest(
        std_vel=std_vel, std_accel=0.05, std_steer=np.radians(1.e-4), std_theta=std_theta, std_dtheta=std_dtheta)
    rob1.P = np.diag([.0, .1, .1, np.radians(0.001), 0.0])
    rob2.P = np.diag([.0, .1, .1, np.radians(0.001), 0.0])

    X = rob1.x
    X1 = rob2.x

    for u, vel_mes, theta_e, dtheta_e in zip(U_a.T, vel_err.tolist()[0], theta_err.tolist()[0], dtheta_err.tolist()[0]):

        u = u.T
        # rob1.predict(u)
        # rob1.update(array([[vel_mes], [dtheta_e]]), R=matrix([[std_vel**2, 0], [0, std_dtheta**2]]))
        X = np.concatenate((X, rob1.x), axis=1)
        print(u)
        rob2.predict(u)
        rob2.update(array([[vel_mes], [dtheta_e]]), R=matrix([[std_vel**2, 0], [0, std_dtheta**2]]))
        X1 = np.concatenate((X1, rob2.x), axis=1)

    plt.figure()
    plt.title('Velocity')
    plt.plot(X[2, :].tolist()[0], '--r')
    plt.plot(State[2, :].tolist()[0], 'b')
    plt.plot(X1[2, :].tolist()[0], '--g')

    plt.figure()
    plt.title('Oriantation')
    plt.plot(X[3, :].tolist()[0], '--r')
    plt.plot(State[3, :].tolist()[0], 'b')
    plt.plot(X1[3, :].tolist()[0], '--g')

    plt.figure()
    plt.title('Angular velocity')
    plt.plot(X[4, :].tolist()[0], '--r')
    plt.plot(State[4, :].tolist()[0], 'b')
    plt.plot(dtheta_err.tolist()[0], '--g')
    plt.plot(X1[4, :].tolist()[0], '-^g')
    plt.legend(['EKF', 'Real', 'Error'])

    plt.figure()
    plt.plot(X[0, :].tolist()[0], X[1, :].tolist()[0], '--r')
    plt.plot(State[0, :].tolist()[0], State[1, :].tolist()[0], 'b')
    plt.plot(X1[0, :].tolist()[0], X1[1, :].tolist()[0], '--g')
    plt.show()


if __name__ == '__main__':
    main()
