from filterpy.kalman import ExtendedKalmanFilter as EKF
from numpy import dot, array, sqrt, matrix, cos, sin, tan
import sympy
import math
import numpy as np
from sympy import symbols, Matrix
from matplotlib import pyplot as plt

import plotHelp


class RobotEKF(EKF):
    def __init__(self, dt, wheelbase, std_acc, std_steer):
        EKF.__init__(self, 5, 2, 2)
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

        if alpha < 0.0001:
            alpha = 0.0001
        r = self.wheelbase/np.tan(alpha)
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
        super(RobotEKF, self).update(z=z, HJacobian=RobotEKF.H_of, Hx=RobotEKF.Hx, R=R)
